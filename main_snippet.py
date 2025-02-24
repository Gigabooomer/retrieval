from chat_history import generate_standalone_question

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

# Generate the standalone question
question = generate_standalone_question(chat_context, thread_id, message)
