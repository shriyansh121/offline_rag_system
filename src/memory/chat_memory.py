class SimpleChatMemory:
    """
    A lightweight custom memory class that stores messages
    and behaves like ConversationBufferMemory for our RAG chain.
    """

    def __init__(self):
        self.messages = []

    def load_memory_variables(self, _inputs):
        return {"chat_history": self.messages}

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    @property
    def chat_memory(self):
        """Backwards compatibility attribute."""
        return self
        

def get_memory():
    return SimpleChatMemory()
