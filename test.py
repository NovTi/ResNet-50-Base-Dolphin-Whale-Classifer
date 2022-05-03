import torch
from torch.utils.data import DataLoader
from dataloader import whale_dolphin_test
from BottleNeck import BottleNeck
from ResNet import ResNet50


def main():
    # 1.load the model
    model = ResNet50(BottleNeck, [3, 4, 6, 3], 3, 2)
    model.load_state_dict(torch.load("results/best.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cup')
    model = model.to(device)
    model.eval()

    # 2.load the test data
    root = "dataset/test"
    batch_size = 64
    test_data = whale_dolphin_test(root)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # 3.test
    print("Waiting Test...")
    with torch.no_grad():
        correct = 0
        total = 0
        for (images, labels) in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            _, pre = torch.max(out.data, 1)
            total += labels.size(0)   # why use (0) not [0]???
            correct += (pre == labels).sum().item()
        print(f"The Accuracy is: {correct/total}")


if __name__ == '__main__':
    main()
