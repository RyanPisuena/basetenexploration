"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        # Load model here and assign to self._model.
        self._model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="cuda",
            torch_dtype="auto"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
        )

    def predict(self, request):
        messages = request.pop("messages")
        model_inputs = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(model_inputs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self._model.generate(input_ids=inputs["input_ids"], max_length=256)
        return {"output": self._tokenizer.decode(outputs[0], skip_special_tokens=True)}
