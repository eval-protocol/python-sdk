#!/usr/bin/env python3
"""
MCP-Gym Implementation for τ²-Bench

This module implements the airline, mock, and retail domains for τ²-Bench using the MCP-Gym framework.
It provides all the tools as MCP tools for agent evaluation.
"""

import argparse
import os
from typing import Any, Dict, Optional, List
import json

from mcp.server.fastmcp import Context
from airplane_environment.airline_environment import AirlineEnvironment
from mock_environment.mock_environment import MockEnvironment
from retail_environment.retail_environment import RetailEnvironment

from eval_protocol.mcp import McpGym, EnvironmentAdapter
from eval_protocol.mcp.mcpgym import control_plane_endpoint


class AirlineDomainMcp(McpGym):
    """Airline booking MCP server for τ²-Bench integration"""

    def __init__(self, seed: Optional[int] = None):
        """Initialize Airline MCP-Gym environment."""
        # Use EnvironmentAdapter directly as the default adapter
        default_config = {
            "domain": "airline",
            "max_turns": 20,
        }

        self.adapter = EnvironmentAdapter(env_class=AirlineEnvironment, default_config=default_config)

        super().__init__("airline", self.adapter, seed)

    def _register_tools(self):
        """Register airline-specific MCP tools matching τ²-Bench schemas"""

        @self.mcp.tool(name="book_reservation", description="Book a reservation.")
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
            ctx: Context,
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
                        "insurance": insurance,
                    },
                },
            )

        @self.mcp.tool(
            name="calculate",
            description="Calculate the result of a mathematical expression.",
        )
        def calculate(expression: str, ctx: Context) -> Dict[str, Any]:
            """Calculate mathematical expressions"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {"action": "calculate", "parameters": {"expression": expression}},
            )

        @self.mcp.tool(name="cancel_reservation", description="Cancel the whole reservation.")
        def cancel_reservation(reservation_id: str, ctx: Context) -> Dict[str, Any]:
            """Cancel a reservation"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "cancel_reservation",
                    "parameters": {"reservation_id": reservation_id},
                },
            )

        @self.mcp.tool(
            name="get_reservation_details",
            description="Get the details of a reservation.",
        )
        def get_reservation_details(reservation_id: str, ctx: Context) -> Dict[str, Any]:
            """Get reservation details"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "get_reservation_details",
                    "parameters": {"reservation_id": reservation_id},
                },
            )

        @self.mcp.tool(
            name="get_user_details",
            description="Get the details of a user, including their reservations.",
        )
        def get_user_details(user_id: str, ctx: Context) -> Dict[str, Any]:
            """Get user details"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {"action": "get_user_details", "parameters": {"user_id": user_id}},
            )

        @self.mcp.tool(
            name="list_all_airports",
            description="Returns a list of all available airports.",
        )
        def list_all_airports(ctx: Context) -> Dict[str, Any]:
            """List all available airports"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "list_all_airports", "parameters": {}}
            )

        @self.mcp.tool(
            name="search_direct_flight",
            description="Search for direct flights between two cities on a specific date.",
        )
        def search_direct_flight(origin: str, destination: str, date: str, ctx: Context) -> Dict[str, Any]:
            """Search for direct flights"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "search_direct_flight",
                    "parameters": {
                        "origin": origin,
                        "destination": destination,
                        "date": date,
                    },
                },
            )

        @self.mcp.tool(
            name="search_onestop_flight",
            description="Search for one-stop flights between two cities on a specific date.",
        )
        def search_onestop_flight(origin: str, destination: str, date: str, ctx: Context) -> Dict[str, Any]:
            """Search for one-stop flights"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "search_onestop_flight",
                    "parameters": {
                        "origin": origin,
                        "destination": destination,
                        "date": date,
                    },
                },
            )

        @self.mcp.tool(
            name="send_certificate",
            description="Send a certificate to a user. Be careful!",
        )
        def send_certificate(user_id: str, amount: int, ctx: Context) -> Dict[str, Any]:
            """Send a certificate to a user"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "send_certificate",
                    "parameters": {"user_id": user_id, "amount": amount},
                },
            )

        @self.mcp.tool(
            name="transfer_to_human_agents",
            description="Transfer the user to a human agent, with a summary of the user's issue.",
        )
        def transfer_to_human_agents(summary: str, ctx: Context) -> Dict[str, Any]:
            """Transfer to human agent"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "transfer_to_human_agents",
                    "parameters": {"summary": summary},
                },
            )

        @self.mcp.tool(
            name="update_reservation_baggages",
            description="Update the baggage information of a reservation.",
        )
        def update_reservation_baggages(
            reservation_id: str,
            total_baggages: int,
            nonfree_baggages: int,
            payment_id: str,
            ctx: Context,
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
                        "payment_id": payment_id,
                    },
                },
            )

        @self.mcp.tool(
            name="update_reservation_flights",
            description="Update the flight information of a reservation.",
        )
        def update_reservation_flights(
            reservation_id: str,
            cabin: str,
            flights: List[Dict[str, Any]],
            payment_id: str,
            ctx: Context,
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
                        "payment_id": payment_id,
                    },
                },
            )

        @self.mcp.tool(
            name="update_reservation_passengers",
            description="Update the passenger information of a reservation.",
        )
        def update_reservation_passengers(
            reservation_id: str, passengers: List[Dict[str, Any]], ctx: Context
        ) -> Dict[str, Any]:
            """Update reservation passenger information"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "update_reservation_passengers",
                    "parameters": {
                        "reservation_id": reservation_id,
                        "passengers": passengers,
                    },
                },
            )

        @self.mcp.tool(name="get_flight_status", description="Get the status of a flight.")
        def get_flight_status(flight_number: str, date: str, ctx: Context) -> Dict[str, Any]:
            """Get flight status"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "get_flight_status",
                    "parameters": {"flight_number": flight_number, "date": date},
                },
            )

    @staticmethod
    def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
        """Return observation as JSON-serialisable dict.

        For the airline domain the environment already returns dictionaries, so
        we simply pass them through.  If another type slips through, wrap it.
        """
        if isinstance(obs, dict):
            return obs
        return {"observation": obs}


class MockDomainMcp(McpGym):
    """Mock domain MCP server for τ²-Bench integration"""

    def __init__(self, seed: Optional[int] = None):
        """Initialize Mock MCP-Gym environment."""
        # Use EnvironmentAdapter directly as the default adapter
        default_config = {
            "domain": "mock",
            "max_turns": 20,
        }

        self.adapter = EnvironmentAdapter(env_class=MockEnvironment, default_config=default_config)

        super().__init__("mock", self.adapter, seed)

    def _register_tools(self):
        """Register mock-specific MCP tools matching τ²-Bench schemas"""

        @self.mcp.tool(name="create_task", description="Create a new task for a user.")
        def create_task(user_id: str, title: str, ctx: Context, description: str = None) -> Dict[str, Any]:
            """Create a new task for a user"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "create_task",
                    "parameters": {"user_id": user_id, "title": title, "description": description},
                },
            )

        @self.mcp.tool(name="get_users", description="Get all users in the database.")
        def get_users(ctx: Context) -> Dict[str, Any]:
            """Get all users in the database"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(session_id, {"action": "get_users", "parameters": {}})

        @self.mcp.tool(name="update_task_status", description="Update the status of a task.")
        def update_task_status(task_id: str, status: str, ctx: Context) -> Dict[str, Any]:
            """Update the status of a task"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "update_task_status", "parameters": {"task_id": task_id, "status": status}}
            )

        @self.mcp.tool(
            name="assert_number_of_tasks", description="Check if the number of tasks for a user is as expected."
        )
        def assert_number_of_tasks(user_id: str, expected_number: int, ctx: Context) -> Dict[str, Any]:
            """Check if the number of tasks for a user is as expected"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "assert_number_of_tasks",
                    "parameters": {"user_id": user_id, "expected_number": expected_number},
                },
            )

        @self.mcp.tool(name="assert_task_status", description="Check if the status of a task is as expected.")
        def assert_task_status(task_id: str, expected_status: str, ctx: Context) -> Dict[str, Any]:
            """Check if the status of a task is as expected"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "assert_task_status",
                    "parameters": {"task_id": task_id, "expected_status": expected_status},
                },
            )

        @self.mcp.tool(
            name="transfer_to_human_agents",
            description=""" Transfer the user to a human agent, with a summary of the user's issue.
            Only transfer if
            -  the user explicitly asks for a human agent
            -  given the policy and the available tools, you cannot solve the user's issue.""",
        )
        def transfer_to_human_agents(summary: str, ctx: Context) -> Dict[str, Any]:
            """Transfer the user to a human agent"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "transfer_to_human_agents", "parameters": {"summary": summary}}
            )

    @staticmethod
    def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
        """Return observation as JSON-serialisable dict.

        For the mock domain the environment already returns dictionaries, so
        we simply pass them through.  If another type slips through, wrap it.
        """
        if isinstance(obs, dict):
            return obs
        return {"observation": obs}


class RetailDomainMcp(McpGym):
    """Retail domain MCP server for τ²-Bench integration"""

    def __init__(self, seed: Optional[int] = None):
        """Initialize Retail MCP-Gym environment."""
        # Use EnvironmentAdapter directly as the default adapter
        default_config = {
            "domain": "retail",
            "max_turns": 20,
        }

        self.adapter = EnvironmentAdapter(env_class=RetailEnvironment, default_config=default_config)

        super().__init__("retail", self.adapter, seed)

    def _register_tools(self):
        """Register retail-specific MCP tools matching τ²-Bench schemas"""

        @self.mcp.tool(name="calculate", description="Calculate the result of a mathematical expression.")
        def calculate(expression: str, ctx: Context) -> Dict[str, Any]:
            """Calculate mathematical expressions"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "calculate", "parameters": {"expression": expression}}
            )

        @self.mcp.tool(
            name="cancel_pending_order",
            description="""Cancel a pending order. If the order is already processed or delivered,
            it cannot be cancelled. The agent needs to explain the cancellation detail
            and ask for explicit user confirmation (yes/no) to proceed. If the user confirms,
            the order status will be changed to 'cancelled' and the payment will be refunded.
            The refund will be added to the user's gift card balance immediately if the payment
            was made using a gift card, otherwise the refund would take 5-7 business days to process.
            The function returns the order details after the cancellation. 
            Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
            reason: The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.""",
        )
        def cancel_pending_order(order_id: str, reason: str, ctx: Context) -> Dict[str, Any]:
            """Cancel a pending order"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "cancel_pending_order", "parameters": {"order_id": order_id, "reason": reason}}
            )

        @self.mcp.tool(
            name="exchange_delivered_order_items",
            description="""Exchange items in a delivered order to new items of the same product type.
            For a delivered order, return or exchange can be only done once by the agent.
            The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.""",
        )
        def exchange_delivered_order_items(
            order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str, ctx: Context
        ) -> Dict[str, Any]:
            """Exchange items in a delivered order"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "exchange_delivered_order_items",
                    "parameters": {
                        "order_id": order_id,
                        "item_ids": item_ids,
                        "new_item_ids": new_item_ids,
                        "payment_method_id": payment_method_id,
                    },
                },
            )

        @self.mcp.tool(
            name="find_user_id_by_name_zip",
            description="""Find user id by first name, last name, and zip code. If the user is not found, the function
            will return an error message. By default, find user id by email, and only call this function
            if the user is not found by email or cannot remember email.""",
        )
        def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str, ctx: Context) -> Dict[str, Any]:
            """Find user id by name and zip"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "find_user_id_by_name_zip",
                    "parameters": {"first_name": first_name, "last_name": last_name, "zip": zip},
                },
            )

        @self.mcp.tool(
            name="find_user_id_by_email",
            description="Find user id by email. If the user is not found, the function will return an error message.",
        )
        def find_user_id_by_email(email: str, ctx: Context) -> Dict[str, Any]:
            """Find user id by email"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "find_user_id_by_email", "parameters": {"email": email}}
            )

        @self.mcp.tool(name="get_order_details", description="Get the status and details of an order.")
        def get_order_details(order_id: str, ctx: Context) -> Dict[str, Any]:
            """Get order details"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "get_order_details", "parameters": {"order_id": order_id}}
            )

        @self.mcp.tool(name="get_product_details", description="Get the inventory details of a product.")
        def get_product_details(product_id: str, ctx: Context) -> Dict[str, Any]:
            """Get product details"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "get_product_details", "parameters": {"product_id": product_id}}
            )

        @self.mcp.tool(name="get_user_details", description="Get the details of a user, including their orders.")
        def get_user_details(user_id: str, ctx: Context) -> Dict[str, Any]:
            """Get user details"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "get_user_details", "parameters": {"user_id": user_id}}
            )

        @self.mcp.tool(
            name="list_all_product_types",
            description="""List the name and product id of all product types.
            Each product type has a variety of different items with unique item ids and options.
            There are only 50 product types in the store.""",
        )
        def list_all_product_types(ctx: Context) -> Dict[str, Any]:
            """List all product types"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "list_all_product_types", "parameters": {}}
            )

        @self.mcp.tool(
            name="modify_pending_order_address",
            description="Modify the shipping address of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        )
        def modify_pending_order_address(
            order_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str, ctx: Context
        ) -> Dict[str, Any]:
            """Modify the shipping address of a pending order"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "modify_pending_order_address",
                    "parameters": {
                        "order_id": order_id,
                        "address1": address1,
                        "address2": address2,
                        "city": city,
                        "state": state,
                        "country": country,
                        "zip": zip,
                    },
                },
            )

        @self.mcp.tool(
            name="modify_pending_order_items",
            description="Modify items in a pending order to new items of the same product type. For a pending order, this function can only be called once. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.",
        )
        def modify_pending_order_items(
            order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str, ctx: Context
        ) -> Dict[str, Any]:
            """Modify items in a pending order"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "modify_pending_order_items",
                    "parameters": {
                        "order_id": order_id,
                        "item_ids": item_ids,
                        "new_item_ids": new_item_ids,
                        "payment_method_id": payment_method_id,
                    },
                },
            )

        @self.mcp.tool(
            name="modify_pending_order_payment",
            description="Modify the payment method of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        )
        def modify_pending_order_payment(order_id: str, payment_method_id: str, ctx: Context) -> Dict[str, Any]:
            """Modify the payment method of a pending order"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "modify_pending_order_payment",
                    "parameters": {"order_id": order_id, "payment_method_id": payment_method_id},
                },
            )

        @self.mcp.tool(
            name="modify_user_address",
            description="Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
        )
        def modify_user_address(
            user_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str, ctx: Context
        ) -> Dict[str, Any]:
            """Modify the default address of a user"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "modify_user_address",
                    "parameters": {
                        "user_id": user_id,
                        "address1": address1,
                        "address2": address2,
                        "city": city,
                        "state": state,
                        "country": country,
                        "zip": zip,
                    },
                },
            )

        @self.mcp.tool(
            name="return_delivered_order_items",
            description="""Return some items of a delivered order.
            The order status will be changed to 'return requested'.
            The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed.
            The user will receive follow-up email for how and where to return the item.""",
        )
        def return_delivered_order_items(
            order_id: str, item_ids: List[str], payment_method_id: str, ctx: Context
        ) -> Dict[str, Any]:
            """Return items from a delivered order"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id,
                {
                    "action": "return_delivered_order_items",
                    "parameters": {"order_id": order_id, "item_ids": item_ids, "payment_method_id": payment_method_id},
                },
            )

        @self.mcp.tool(
            name="transfer_to_human_agents",
            description="""Transfer the user to a human agent, with a summary of the user's issue.
            Only transfer if
            -  the user explicitly asks for a human agent
            -  given the policy and the available tools, you cannot solve the user's issue.
            """,
        )
        def transfer_to_human_agents(summary: str, ctx: Context) -> Dict[str, Any]:
            """Transfer the user to a human agent"""
            session_id = self._get_session_id(ctx)

            return self._execute_session_environment_step(
                session_id, {"action": "transfer_to_human_agents", "parameters": {"summary": summary}}
            )

    @staticmethod
    def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
        """Return observation as JSON-serialisable dict.

        For the retail domain the environment already returns dictionaries, so
        we simply pass them through.  If another type slips through, wrap it.
        """
        if isinstance(obs, dict):
            return obs
        return {"observation": obs}
