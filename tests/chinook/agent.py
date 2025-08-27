from pydantic_ai import Agent, RunContext
import asyncio
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.exceptions import ModelRetry
from db import connect_database


def setup_agent(orchestrator_agent_model: Model):
    connection, cursor, introspection_result = connect_database()

    introspection_result_str = "\n".join([",".join(map(str, item)) for item in introspection_result])

    SYSTEM_PROMPT = f"""You are a helpful assistant that has access to the
Chinook database stored in a Postgres database. You have access to a tool to
execute SQL queries that you should use to answer questions. Your job is to
answer questions about the database. If you run into an error, you should try to
fix the query and try again. Here is the schema of the database:

Schema:
table_name,column_name,data_type,is_nullable
{introspection_result_str}
    """

    agent = Agent(
        system_prompt=SYSTEM_PROMPT,
        model=orchestrator_agent_model,
    )

    @agent.tool(retries=5)
    def execute_sql(ctx: RunContext, query: str) -> dict:
        try:
            cursor.execute(query)
            # Get column headers from cursor description
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            # Get data rows
            rows = cursor.fetchall()

            if not columns or not rows:
                return "No results found."

            # Create markdown table
            table_lines = []

            # Header row
            table_lines.append("| " + " | ".join(columns) + " |")

            # Separator row
            table_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

            # Data rows
            for row in rows:
                # Convert all values to strings and escape pipes
                formatted_row = [str(cell).replace("|", "\\|") if cell is not None else "" for cell in row]
                table_lines.append("| " + " | ".join(formatted_row) + " |")

            return "\n".join(table_lines)
        except Exception as e:
            connection.rollback()
            raise ModelRetry("Please try again with a different query. Here is the error: " + str(e))

    return agent


async def main():
    model = OpenAIModel(
        model="accounts/fireworks/models/kimi-k2-instruct",
        provider="fireworks",
    )
    agent = setup_agent(model)
    result = await agent.run("What is the total number of tracks in the database?")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
