if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix, classification_report
    import torch
    from petclassifier_pytorch import NeuralNetwork, device, test_dataloader, test_dataset

    all_preds = []
    all_labels = []

    model = NeuralNetwork()
    model.load_state_dict(torch.load("model_weights.pth", weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
