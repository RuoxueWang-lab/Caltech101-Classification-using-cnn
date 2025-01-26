import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

from network import CNN

if __name__ == "__main__":
    
    # Preparing data:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 10
    
    dataset = torchvision.datasets.Caltech101(root='./data', download=False, transform=transform)
    dataset_path = "./data/caltech101/101_ObjectCategories/"
    classes = os.listdir(dataset_path)
    classes = [class_name for class_name in classes]
    
    test_size = int(0.2 * len(dataset))
    _, test_set = torch.utils.data.random_split(dataset, [len(dataset)-test_size, test_size])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shufﬂe=False, num_workers=2)
    
    dataiter = iter(test_loader)
    
    # Load saved training model:
    model = CNN()
    model.load_state_dict(torch.load('saved_model.pt', weights_only=True))
    
    # Inferecne
    images, labels = next(dataiter)
    print('Ground-truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predictions[j]] for j in range(batch_size)))

    # save to images 
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/ 2*255+.5*255).permute(1,2,0).numpy().astype('uint8')) 
    im.save("test_pt_images.jpg") 
    print('test_pt_images.jpg saved.')
    
    # Accuracy:
    correct = (predictions==labels).sum().item()
    accuracy = 100 * correct / len(labels)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    
    
    
    
    

