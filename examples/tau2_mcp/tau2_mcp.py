#!/usr/bin/env python3
"""
Airline MCP-Gym Implementation for τ²-Bench

This module implements the airline domain for τ²-Bench using the MCP-Gym framework.
It provides all the airline booking tools as MCP tools for agent evaluation.
"""

import argparse
import os
from typing import Any, Dict, Optional, List
import json

from tau2_adapter import AirlineAdapter
from mcp.server.fastmcp import Context

from reward_kit.mcp import McpGym
from reward_kit.mcp.mcpgym import control_plane_endpoint


class AirlineMcp(McpGym):
    """Airline booking MCP server for τ²-Bench integration"""

    def __init__(self, seed: Optional[int] = None):
        """Initialize Airline MCP-Gym environment."""
        self.adapter = AirlineAdapter()
        super().__init__("airline", self.adapter, seed)

    def _register_tools(self):
        """Register airline-specific MCP tools matching τ²-Bench schemas"""

        @self.mcp.tool(
            name="book_reservation",
            description="Book a reservation."
        )
        def book_reservation(
            user_id: str,
            origin: str,
            destination: str,
            flight_type: str,  # "round_trip" or "one_way"
            cabin: str,  # "business", "economy", "basic_economy"
            flights: List[Dict[str, Any]],
            passengers: List[Dict[str, Any]],
            payment_methods: List[Dict[str, Any]],
            total_baggages: int,
            nonfree_baggages: int,
            insurance: str,  # "yes" or "no"
            ctx: Context
        ) -> Dict[str, Any]:
            """Book a new reservation with all details"""
            session_id = self._get_session_id(ctx)
            session_data = self._get_or_create_session(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "book_reservation",
                    "parameters": {
                        "user_id": user_id,
                        "origin": origin,
                        "destination": destination,
                        "flight_type": flight_type,
                        "cabin": cabin,
                        "flights": flights,
                        "passengers": passengers,
                        "payment_methods": payment_methods,
                        "total_baggages": total_baggages,
                        "nonfree_baggages": nonfree_baggages,
                        "insurance": insurance
                    }
                }
            )

        @self.mcp.tool(
            name="calculate",
            description="Calculate the result of a mathematical expression."
        )
        def calculate(expression: str, ctx: Context) -> Dict[str, Any]:
            """Calculate mathematical expressions"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "calculate",
                    "parameters": {"expression": expression}
                }
            )

        @self.mcp.tool(
            name="cancel_reservation",
            description="Cancel the whole reservation."
        )
        def cancel_reservation(reservation_id: str, ctx: Context) -> Dict[str, Any]:
            """Cancel a reservation"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "cancel_reservation",
                    "parameters": {"reservation_id": reservation_id}
                }
            )

        @self.mcp.tool(
            name="get_reservation_details",
            description="Get the details of a reservation."
        )
        def get_reservation_details(reservation_id: str, ctx: Context) -> Dict[str, Any]:
            """Get reservation details"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "get_reservation_details",
                    "parameters": {"reservation_id": reservation_id}
                }
            )

        @self.mcp.tool(
            name="get_user_details",
            description="Get the details of a user, including their reservations."
        )
        def get_user_details(user_id: str, ctx: Context) -> Dict[str, Any]:
            """Get user details"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "get_user_details",
                    "parameters": {"user_id": user_id}
                }
            )

        @self.mcp.tool(
            name="list_all_airports",
            description="Returns a list of all available airports."
        )
        def list_all_airports(ctx: Context) -> Dict[str, Any]:
            """List all available airports"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "list_all_airports",
                    "parameters": {}
                }
            )

        @self.mcp.tool(
            name="search_direct_flight",
            description="Search for direct flights between two cities on a specific date."
        )
        def search_direct_flight(
            origin: str,
            destination: str,
            date: str,
            ctx: Context
        ) -> Dict[str, Any]:
            """Search for direct flights"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "search_direct_flight",
                    "parameters": {
                        "origin": origin,
                        "destination": destination,
                        "date": date
                    }
                }
            )

        @self.mcp.tool(
            name="search_onestop_flight",
            description="Search for one-stop flights between two cities on a specific date."
        )
        def search_onestop_flight(
            origin: str,
            destination: str,
            date: str,
            ctx: Context
        ) -> Dict[str, Any]:
            """Search for one-stop flights"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "search_onestop_flight",
                    "parameters": {
                        "origin": origin,
                        "destination": destination,
                        "date": date
                    }
                }
            )

        @self.mcp.tool(
            name="send_certificate",
            description="Send a certificate to a user. Be careful!"
        )
        def send_certificate(
            user_id: str,
            amount: int,
            ctx: Context
        ) -> Dict[str, Any]:
            """Send a certificate to a user"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "send_certificate",
                    "parameters": {
                        "user_id": user_id,
                        "amount": amount
                    }
                }
            )

        @self.mcp.tool(
            name="transfer_to_human_agents",
            description="Transfer the user to a human agent, with a summary of the user's issue."
        )
        def transfer_to_human_agents(summary: str, ctx: Context) -> Dict[str, Any]:
            """Transfer to human agent"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "transfer_to_human_agents",
                    "parameters": {"summary": summary}
                }
            )

        @self.mcp.tool(
            name="update_reservation_baggages",
            description="Update the baggage information of a reservation."
        )
        def update_reservation_baggages(
            reservation_id: str,
            total_baggages: int,
            nonfree_baggages: int,
            payment_id: str,
            ctx: Context
        ) -> Dict[str, Any]:
            """Update reservation baggage information"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "update_reservation_baggages",
                    "parameters": {
                        "reservation_id": reservation_id,
                        "total_baggages": total_baggages,
                        "nonfree_baggages": nonfree_baggages,
                        "payment_id": payment_id
                    }
                }
            )

        @self.mcp.tool(
            name="update_reservation_flights",
            description="Update the flight information of a reservation."
        )
        def update_reservation_flights(
            reservation_id: str,
            cabin: str,
            flights: List[Dict[str, Any]],
            payment_id: str,
            ctx: Context
        ) -> Dict[str, Any]:
            """Update reservation flight information"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "update_reservation_flights",
                    "parameters": {
                        "reservation_id": reservation_id,
                        "cabin": cabin,
                        "flights": flights,
                        "payment_id": payment_id
                    }
                }
            )

        @self.mcp.tool(
            name="update_reservation_passengers",
            description="Update the passenger information of a reservation."
        )
        def update_reservation_passengers(
            reservation_id: str,
            passengers: List[Dict[str, Any]],
            ctx: Context
        ) -> Dict[str, Any]:
            """Update reservation passenger information"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "update_reservation_passengers",
                    "parameters": {
                        "reservation_id": reservation_id,
                        "passengers": passengers
                    }
                }
            )

        @self.mcp.tool(
            name="get_flight_status",
            description="Get the status of a flight."
        )
        def get_flight_status(
            flight_number: str,
            date: str,
            ctx: Context
        ) -> Dict[str, Any]:
            """Get flight status"""
            session_id = self._get_session_id(ctx)
            
            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "get_flight_status",
                    "parameters": {
                        "flight_number": flight_number,
                        "date": date
                    }
                }
            )
