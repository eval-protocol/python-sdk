#!/usr/bin/env python3
"""
Airline Environment for τ²-Bench Integration

This module implements an AirlineEnvironment that integrates the τ²-Bench simulation
pattern (Agent/User/Environment communication) with the MCP-Gym framework.
"""
from copy import deepcopy
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import os
from pathlib import Path

from tau2.domains.airline.data_model import (
    AirportCode,
    CabinClass,
    Certificate,
    DirectFlight,
    Flight,
    FlightDateStatus,
    FlightDateStatusAvailable,
    FlightDB,
    FlightType,
    Insurance,
    Passenger,
    Payment,
    Reservation,
    ReservationFlight,
    User,
    FlightInfo,
)

logger = logging.getLogger(__name__)

AIRLINE_DB_PATH = Path(__file__).parent / "db.json"

class AirlineEnvironment:
    """
    Airline environment that integrates τ²-Bench simulation pattern
    with MCP-Gym framework.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db = FlightDB.load(AIRLINE_DB_PATH)
        self.airline_tools = AirlineTools(self.db)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state"""
        # TODO
        
        return {}, {}
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Perform one step of the τ²-Bench simulation.
        """

        action_name = action.get("action", "")
        parameters = action.get("parameters", {})
        
        result = self._execute_airline_action(action_name, parameters)

        # In tau2-bench, if there's a simulated user, the agent cannot terminate the rollout, and there are no per step rewards.
        
        return result, 0.0, False, False, {}

    
    def _calculate_reward(self):
        """Calculate the reward for the entire conversation."""
        pass
    
    def close(self):
        """Clean up environment resources"""
        pass

    def _execute_airline_action(self, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action using airline tools."""
        action_map = {
            "book_reservation": self.airline_tools.book_reservation,
            "cancel_reservation": self.airline_tools.cancel_reservation,
            "get_reservation_details": self.airline_tools.get_reservation_details,
            "get_user_details": self.airline_tools.get_user_details,
            "list_all_airports": self.airline_tools.list_all_airports,
            "search_direct_flight": self.airline_tools.search_direct_flight,
            "search_onestop_flight": self.airline_tools.search_onestop_flight,
            "send_certificate": self.airline_tools.send_certificate,
            "transfer_to_human_agents": self.airline_tools.transfer_to_human_agents,
            "calculate": self.airline_tools.calculate,
            "get_flight_status": self.airline_tools.get_flight_status,
            "update_reservation_baggages": self.airline_tools.update_reservation_baggages,
            "update_reservation_flights": self.airline_tools.update_reservation_flights,
            "update_reservation_passengers": self.airline_tools.update_reservation_passengers,
        }
        
        if action_name in action_map:
            tool_method = action_map[action_name]
            # Call the tool method with parameters
            if parameters:
                return tool_method(**parameters)
            else:
                return tool_method()
        else:
            return {
                "error": f"Unknown action: {action_name}"
            }

class AirlineTools:
    """All the tools for the airline domain."""

    def __init__(self, db: FlightDB) -> None:
        self.db = db

    def _get_user(self, user_id: str) -> User:
        """Get user from database."""
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return self.db.users[user_id]

    def _get_reservation(self, reservation_id: str) -> Reservation:
        """Get reservation from database."""
        if reservation_id not in self.db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self.db.reservations[reservation_id]

    def _get_flight(self, flight_number: str) -> Flight:
        """Get flight from database."""
        if flight_number not in self.db.flights:
            raise ValueError(f"Flight {flight_number} not found")
        return self.db.flights[flight_number]

    def _get_flight_instance(self, flight_number: str, date: str) -> FlightDateStatus:
        """Get flight instance from database."""
        flight = self._get_flight(flight_number)
        if date not in flight.dates:
            raise ValueError(f"Flight {flight_number} not found on date {date}")
        return flight.dates[date]

    def _get_new_reservation_id(self) -> str:
        """Get a new reservation id."""
        for reservation_id in ["HATHAT", "HATHAU", "HATHAV"]:
            if reservation_id not in self.db.reservations:
                return reservation_id
        raise ValueError("Too many reservations")

    def _get_new_payment_id(self) -> List[int]:
        """Get a new payment id."""
        return [3221322, 3221323, 3221324]

    def _get_datetime(self) -> str:
        """Get the current datetime."""
        return "2024-05-15T15:00:00"

    def _search_direct_flight(
        self,
        date: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        leave_after: Optional[str] = None,
    ) -> List[DirectFlight]:
        """Search for direct flights"""
        results = []
        for flight in self.db.flights.values():
            check = (
                (origin is None or flight.origin == origin)
                and (destination is None or flight.destination == destination)
                and (date in flight.dates)
                and (flight.dates[date].status == "available")
                and (
                    leave_after is None
                    or flight.scheduled_departure_time_est >= leave_after
                )
            )
            if check:
                direct_flight = DirectFlight(
                    flight_number=flight.flight_number,
                    origin=flight.origin,
                    destination=flight.destination,
                    status="available",
                    scheduled_departure_time_est=flight.scheduled_departure_time_est,
                    scheduled_arrival_time_est=flight.scheduled_arrival_time_est,
                    available_seats=flight.dates[date].available_seats,
                    prices=flight.dates[date].prices,
                )
                results.append(direct_flight)
        return results

    def _payment_for_update(
        self, user: User, payment_id: str, total_price: int
    ) -> Optional[Payment]:
        """Process payment for update reservation"""
        if payment_id not in user.payment_methods:
            raise ValueError("Payment method not found")
        payment_method = user.payment_methods[payment_id]
        if payment_method.source == "certificate":
            raise ValueError("Certificate cannot be used to update reservation")
        elif (
            payment_method.source == "gift_card" and payment_method.amount < total_price
        ):
            raise ValueError("Gift card balance is not enough")

        if payment_method.source == "gift_card":
            payment_method.amount -= total_price

        payment = None
        if total_price != 0:
            payment = Payment(
                payment_id=payment_id,
                amount=total_price,
            )
        return payment

    # Tool methods
    def book_reservation(
        self,
        user_id: str,
        origin: str,
        destination: str,
        flight_type: FlightType,
        cabin: CabinClass,
        flights: List[Any],
        passengers: List[Any],
        payment_methods: List[Any],
        total_baggages: int,
        nonfree_baggages: int,
        insurance: Insurance,
    ) -> Dict[str, Any]:
        """Book a reservation."""
        # Convert dict inputs to proper objects
        if flights and isinstance(flights[0], dict):
            flights = [FlightInfo(**flight) for flight in flights]
        if passengers and isinstance(passengers[0], dict):
            passengers = [Passenger(**passenger) for passenger in passengers]
        if payment_methods and isinstance(payment_methods[0], dict):
            payment_methods = [Payment(**payment) for payment in payment_methods]

        user = self._get_user(user_id)
        reservation_id = self._get_new_reservation_id()

        reservation = Reservation(
            reservation_id=reservation_id,
            user_id=user_id,
            origin=origin,
            destination=destination,
            flight_type=flight_type,
            cabin=cabin,
            flights=[],
            passengers=deepcopy(passengers),
            payment_history=deepcopy(payment_methods),
            created_at=self._get_datetime(),
            total_baggages=total_baggages,
            nonfree_baggages=nonfree_baggages,
            insurance=insurance,
        )

        # Update flights and calculate price
        total_price = 0
        all_flights_date_data = []

        for flight_info in flights:
            flight_number = flight_info.flight_number
            flight = self._get_flight(flight_number)
            flight_date_data = self._get_flight_instance(
                flight_number=flight_number, date=flight_info.date
            )
            
            if not isinstance(flight_date_data, FlightDateStatusAvailable):
                raise ValueError(f"Flight {flight_number} not available on date {flight_info.date}")
            
            if flight_date_data.available_seats[cabin] < len(passengers):
                raise ValueError(f"Not enough seats on flight {flight_number}")
            
            price = flight_date_data.prices[cabin]
            reservation.flights.append(
                ReservationFlight(
                    origin=flight.origin,
                    destination=flight.destination,
                    flight_number=flight_number,
                    date=flight_info.date,
                    price=price,
                )
            )
            all_flights_date_data.append(flight_date_data)
            total_price += price * len(passengers)

        # Add fees
        if insurance == "yes":
            total_price += 30 * len(passengers)
        total_price += 50 * nonfree_baggages

        # Validate and process payments
        for payment_method in payment_methods:
            payment_id = payment_method.payment_id
            amount = payment_method.amount
            if payment_id not in user.payment_methods:
                raise ValueError(f"Payment method {payment_id} not found")

            user_payment_method = user.payment_methods[payment_id]
            if user_payment_method.source in {"gift_card", "certificate"}:
                if user_payment_method.amount < amount:
                    raise ValueError(f"Not enough balance in payment method {payment_id}")

        total_payment = sum(payment.amount for payment in payment_methods)
        if total_payment != total_price:
            raise ValueError(f"Payment amount does not add up, total price is {total_price}, but paid {total_payment}")

        # Process payments
        for payment_method in payment_methods:
            payment_id = payment_method.payment_id
            amount = payment_method.amount
            user_payment_method = user.payment_methods[payment_id]
            if user_payment_method.source == "gift_card":
                user_payment_method.amount -= amount
            elif user_payment_method.source == "certificate":
                user.payment_methods.pop(payment_id)

        # Update database
        for flight_date_data in all_flights_date_data:
            flight_date_data.available_seats[cabin] -= len(passengers)
        self.db.reservations[reservation_id] = reservation
        self.db.users[user_id].reservations.append(reservation_id)
        
        return {"reservation": reservation.model_dump()}

    def cancel_reservation(self, reservation_id: str) -> Dict[str, Any]:
        """Cancel the whole reservation."""
        reservation = self._get_reservation(reservation_id)
        
        # Reverse the payment
        refunds = []
        for payment in reservation.payment_history:
            refunds.append(Payment(payment_id=payment.payment_id, amount=-payment.amount))
        reservation.payment_history.extend(refunds)
        reservation.status = "cancelled"
        
        return {"reservation": reservation.model_dump()}

    def get_reservation_details(self, reservation_id: str) -> Dict[str, Any]:
        """Get the details of a reservation."""
        reservation = self._get_reservation(reservation_id)
        return {"reservation": reservation.model_dump()}

    def get_user_details(self, user_id: str) -> Dict[str, Any]:
        """Get the details of a user."""
        user = self._get_user(user_id)
        return {"user": user.model_dump()}

    def list_all_airports(self) -> Dict[str, Any]:
        """Returns a list of all available airports."""
        airports = [
            AirportCode(iata="SFO", city="San Francisco"),
            AirportCode(iata="JFK", city="New York"),
            AirportCode(iata="LAX", city="Los Angeles"),
            AirportCode(iata="ORD", city="Chicago"),
            AirportCode(iata="DFW", city="Dallas"),
            AirportCode(iata="DEN", city="Denver"),
            AirportCode(iata="SEA", city="Seattle"),
            AirportCode(iata="ATL", city="Atlanta"),
            AirportCode(iata="MIA", city="Miami"),
            AirportCode(iata="BOS", city="Boston"),
            AirportCode(iata="PHX", city="Phoenix"),
            AirportCode(iata="IAH", city="Houston"),
            AirportCode(iata="LAS", city="Las Vegas"),
            AirportCode(iata="MCO", city="Orlando"),
            AirportCode(iata="EWR", city="Newark"),
            AirportCode(iata="CLT", city="Charlotte"),
            AirportCode(iata="MSP", city="Minneapolis"),
            AirportCode(iata="DTW", city="Detroit"),
            AirportCode(iata="PHL", city="Philadelphia"),
            AirportCode(iata="LGA", city="LaGuardia"),
        ]
        return {"airports": airports}

    def search_direct_flight(self, origin: str, destination: str, date: str) -> Dict[str, Any]:
        """Search for direct flights between two cities on a specific date."""
        flights = self._search_direct_flight(date=date, origin=origin, destination=destination)
        return {"flights": [flight.model_dump() for flight in flights]}

    def search_onestop_flight(self, origin: str, destination: str, date: str) -> Dict[str, Any]:
        """Search for one-stop flights between two cities on a specific date."""
        results = []
        for result1 in self._search_direct_flight(date=date, origin=origin, destination=None):
            result1.date = date
            date2 = f"2024-05-{int(date[-2:]) + 1}" if "+1" in result1.scheduled_arrival_time_est else date
            
            for result2 in self._search_direct_flight(
                date=date2,
                origin=result1.destination,
                destination=destination,
                leave_after=result1.scheduled_arrival_time_est,
            ):
                result2.date = date2
                results.append([result1.model_dump(), result2.model_dump()])
        return {"flights": results}

    def send_certificate(self, user_id: str, amount: int) -> Dict[str, Any]:
        """Send a certificate to a user."""
        user = self._get_user(user_id)
        
        for payment_id in [f"certificate_{id}" for id in self._get_new_payment_id()]:
            if payment_id not in user.payment_methods:
                new_payment = Certificate(
                    id=payment_id,
                    amount=amount,
                    source="certificate",
                )
                user.payment_methods[payment_id] = new_payment
                return {"message": f"Certificate {payment_id} added to user {user_id} with amount {amount}."}
        raise ValueError("Too many certificates")

    def transfer_to_human_agents(self, summary: str) -> Dict[str, Any]:
        """Transfer the user to a human agent."""
        return {"message": "Transfer successful", "summary": summary}

    def calculate(self, expression: str) -> Dict[str, Any]:
        """Calculate the result of a mathematical expression."""
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        result = str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        return {"result": result}

    def get_flight_status(self, flight_number: str, date: str) -> Dict[str, Any]:
        """Get the status of a flight."""
        status = self._get_flight_instance(flight_number, date).status
        return {"status": status}

    def update_reservation_baggages(
        self, reservation_id: str, total_baggages: int, nonfree_baggages: int, payment_id: str
    ) -> Dict[str, Any]:
        """Update the baggage information of a reservation."""
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)
            
        total_price = 50 * max(0, nonfree_baggages - reservation.nonfree_baggages)
        payment = self._payment_for_update(user, payment_id, total_price)
        
        if payment is not None:
            reservation.payment_history.append(payment)
        
        reservation.total_baggages = total_baggages
        reservation.nonfree_baggages = nonfree_baggages
        
        return {"reservation": reservation.model_dump()}

    def update_reservation_flights(
        self, reservation_id: str, cabin: str, flights: List[Any], payment_id: str
    ) -> Dict[str, Any]:
        """Update the flight information of a reservation."""
        if flights and isinstance(flights[0], dict):
            flights = [FlightInfo(**flight) for flight in flights]
            
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)
        
        total_price = 0
        reservation_flights = []
        
        for flight_info in flights:
            matching_reservation_flight = next(
                (
                    reservation_flight
                    for reservation_flight in reservation.flights
                    if reservation_flight.flight_number == flight_info.flight_number
                    and reservation_flight.date == flight_info.date
                    and cabin == reservation.cabin
                ),
                None,
            )
            
            if matching_reservation_flight:
                total_price += matching_reservation_flight.price * len(reservation.passengers)
                reservation_flights.append(matching_reservation_flight)
                continue
            
            flight = self._get_flight(flight_info.flight_number)
            flight_date_data = self._get_flight_instance(flight_info.flight_number, flight_info.date)
            
            if not isinstance(flight_date_data, FlightDateStatusAvailable):
                raise ValueError(f"Flight {flight_info.flight_number} not available on date {flight_info.date}")
            
            if flight_date_data.available_seats[cabin] < len(reservation.passengers):
                raise ValueError(f"Not enough seats on flight {flight_info.flight_number}")
            
            reservation_flight = ReservationFlight(
                flight_number=flight_info.flight_number,
                date=flight_info.date,
                price=flight_date_data.prices[cabin],
                origin=flight.origin,
                destination=flight.destination,
            )
            total_price += reservation_flight.price * len(reservation.passengers)
            reservation_flights.append(reservation_flight)
        
        total_price -= sum(flight.price for flight in reservation.flights) * len(reservation.passengers)
        payment = self._payment_for_update(user, payment_id, total_price)
        
        if payment is not None:
            reservation.payment_history.append(payment)
        
        reservation.flights = reservation_flights
        reservation.cabin = cabin
        
        return {"reservation": reservation.model_dump()}

    def update_reservation_passengers(
        self, reservation_id: str, passengers: List[Any]
    ) -> Dict[str, Any]:
        """Update the passenger information of a reservation."""
        if passengers and isinstance(passengers[0], dict):
            passengers = [Passenger(**passenger) for passenger in passengers]
            
        reservation = self._get_reservation(reservation_id)
            
        if len(passengers) != len(reservation.passengers):
            raise ValueError("Number of passengers does not match")
        
        reservation.passengers = deepcopy(passengers)
        return {"reservation": reservation.model_dump()}
