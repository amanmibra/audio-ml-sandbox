import torch
from train import CNNetwork, download_mnist_datasets

from tqdm import tqdm

INFERENCE_LIMIT = 100

CLASS_MAPPING = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def predict(model, x, y, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(x)

        predicted_indx = predictions[0].argmax(0)
        predicted = class_mapping[predicted_indx]
        expected = class_mapping[y]

    return predicted, expected

if __name__ == "__main__":
    # load model
    model = CNNetwork()
    state_dict = torch.load("mnist/mnist_model.pth")
    model.load_state_dict(state_dict)

    # load MNIST dataset
    _, validation_data = download_mnist_datasets()

    total = INFERENCE_LIMIT
    correct = 0
    failed = []

    for i in tqdm(range(INFERENCE_LIMIT), "Validating model..."):
        x, y = validation_data[i][0], validation_data[i][1]

        # get prediction and actual value
        pred, expected = predict(model, x, y, CLASS_MAPPING)

        if pred == expected:
            correct += 1
        else:
            failed.append(i)

        print(f"[{i}]: Prediction: {pred}, Expected: {expected}")
    print("---------------------------------------")
    print(f"Validation Outcome: {correct}/{total}")
    if failed:
        print(f"Failed tests: {failed}")


