# abstractions
class LLMRuntime(ABC):

    def load():
        pass

    def unload():
        pass

    def generate():
        pass

    def stream():
        pass

# implementation
LlamaCppRuntime
TransformersRuntime
VllmRuntime
class MyRuntime(LLMRuntime):
    ...

# usage
runtime = RuntimeFactory.create(model)