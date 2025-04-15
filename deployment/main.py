import argparse
from typing import Dict
import numpy as np
import pickle
import joblib
from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse
from kserve.utils.utils import generate_uuid
import logging
logger = logging.getLogger(__name__)


class MyModel(Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.vectorizer = None
        self.ready = False
        self.load()

    def load(self):
        # Load feature vectorizer and trained model
        try:
            self.vectorizer = joblib.load("./saved_model/vectorizer_v1.joblib")
            self.model = joblib.load("./saved_model/model_v1.joblib")


            self.ready = True
            logger.info("model and vectorizer loaded")
        except Exception as e:
            logger.error(f"error loading model and vectorizer: {e}")
            
    def preprocess(self, payload: InferRequest, headers: Dict[str, str] = None, **kwargs) -> np.ndarray:
        try:
            infer_input = payload["instances"]
            raw_data = np.array(infer_input)
           
            print(f"Received input: {raw_data}")
            vectorized_data = self.vectorizer.transform(raw_data)
            return vectorized_data
        except Exception as e:
            logger.error(f"Preprocess error: {e}")
            raise


    def predict(self, data: np.ndarray, headers: Dict[str, str] = None, **kwargs) -> InferResponse:

            result = self.model.predict(data)
            result_bytes = np.array(result, dtype=object)  # Use object dtype for strings

            response_id = generate_uuid()
            infer_output = InferOutput(
                name="output-0",
                shape=list(result_bytes.shape),
                datatype="BYTES",
                data=result_bytes.tolist()  # Convert to list for KServe
            )
            infer_response = InferResponse(
                model_name=self.name,
                infer_outputs=[infer_output],
                response_id=response_id
            )
            return result_bytes.tolist()


    # def postprocess(self, payload, **kwargs):
    #     # Optionally postprocess payload
    #     return payload


parser = argparse.ArgumentParser(parents=[model_server.parser])

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel("model")
    ModelServer().start([model])
