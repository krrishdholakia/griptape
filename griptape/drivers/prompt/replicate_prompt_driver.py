import anthropic
from attr import define, field, Factory
from griptape.artifacts import TextArtifact
from griptape.drivers import BasePromptDriver
from litellm import completion 

@define
class ReplicatePromptDriver(BasePromptDriver):
    api_key: str = field(kw_only=True)
    model: str = field(default="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1", kw_only=True)

    def try_run(self, value: str) -> TextArtifact:
        return self.__run_completion(value)

    def __run_completion(self, value: str) -> TextArtifact:
        messages=[{"role": "user","content": value}]
        # LiteLLM takes the same params as the OpenAI Python SDK https://github.com/BerriAI/litellm
        response = completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key
        )

        return TextArtifact(value=response['choices'][0]['message']['content'])
