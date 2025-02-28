import chainlit as cl
import asyncio  # To simulate async tasks

@cl.on_message
async def main(message: cl.Message):
    cl.log("Received user message")  # Appears in Chainlit UI logs
    print("Received user message")   # Appears in VS Code terminal

    # Step 1: Validate Input
    async with cl.Step("Step 1: Validating input..."):
        await asyncio.sleep(1)  # Simulating delay
        cl.log("Input validation completed")
        print("Step 1 completed")

    # Step 2: Fetch Data
    async with cl.Step("Step 2: Fetching relevant data..."):
        await asyncio.sleep(2)  # Simulating async data fetching
        cl.log("Data fetched successfully")
        print("Step 2 completed")

    # Step 3: Process Data
    async with cl.Step("Step 3: Processing data..."):
        await asyncio.sleep(1)  # Simulating data processing
        cl.log("Data processing completed")
        print("Step 3 completed")

    # Step 4: Generate Response
    async with cl.Step("Step 4: Generating response..."):
        await asyncio.sleep(1)  # Simulating response generation
        response_text = "Here is your answer!"
        cl.log("Response generated")
        print("Step 4 completed")

    # Step 5: Send Response
    async with cl.Step("Step 5: Sending response..."):
        await cl.Message(response_text).send()
        cl.log("Response sent to user")
        print("Step 5 completed")
