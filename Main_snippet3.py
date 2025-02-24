from chat_history import generate_standalone_question

async def respond(self, input: InputModel):

    message = input["message"]
    user_id = input["user_id"]
    role = input["role"]
    thread_id = input["thread_id"]
    message_id = input["message_id"]
    features = input["features"]

    intermediate_steps = {}
    fixed_error = False

    try:
        logger.info(f"Received prompt: {message}")

        # Initialize Step based on run mode
        if self.run_mode == "chainlit":
            import chainlit as cl
            Step = cl.Step
        else:
            Step = DummyStep

        # Detect Language
        async with Step(name="Language") as step:
            lang = detect(message)
            logger.info(f"Detected Language Code: {lang}")

            lang = lang_dict.get(lang, "English")  # Default to English if not mapped
            logger.info(f"Mapped Language: {lang}")

            step.output = lang

        # Retrieve chat history from database
        if self.include_chat_history:
            async with Step(name="Standalone Question") as step:
                if self.use_standalone_question:
                    async with SessionFactoryLogging() as session:
                        prev_messages = await get_messages_by_thread_id_(
                            session=session,
                            thread_id=thread_id,
                            order="desc",
                            limit=self.memory_window
                        )

                        chat_context = [
                            {"user": d["message"], "assistant": d["response"]}
                            for d in reversed(prev_messages)
                        ]

                    # Generate the standalone question using run_chain
                    question = run_chain(
                        self.standalone_question_chain,
                        dict(
                            question=message,
                            context=generate_standalone_question(chat_context, message, self.standalone_question_chain, self.schema),
                            schema=self.schema
                        )
                    )

                    intermediate_steps["standalone_question"] = question
                    step.output = question

    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        raise RuntimeError("An error occurred during the chat history retrieval process.")
