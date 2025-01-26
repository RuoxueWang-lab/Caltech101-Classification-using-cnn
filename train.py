import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import os

from network import CNN

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 32
    
    dataset = torchvision.datasets.Caltech101(root='./data', download=True, transform=transform)
    dataset_path = './data/caltech101/101_ObjectCategories'
    classes = sorted(os.listdir(dataset_path))
    classes = [name for name in classes ]
    
    train_size = int(0.8 * len(dataset))
    train_set, _ = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shufﬂe=True, num_workers=2)
    
    # An exmple batch of images to help us fine-tune and debug
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/ 2*255+.5*255).permute(1,2,0).numpy().astype('uint8')) 
    im.save("train_pt_images.jpg") 
    print('train_pt_images.jpg saved.')
    print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    
    cnn = CNN()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimisor = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(3):
        running_loss = 0
        for num_batch, data in enumerate(train_loader, 0):
            print(f'numbatch:{num_batch}')
            iuputs, labels = data
            optimisor.zero_grad()
            outputs = cnn(iuputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimisor.step()
            
            running_loss += loss.item()
            
            # print statistics
            if num_batch % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, num_batch + 1, running_loss / 2000))
                running_loss = 0.0
        print(f'Epoch:{epoch+1}/3')
    print('Training done.')
    
    torch.save(cnn.state_dict(), 'saved_model.pt') 
    print('Model saved.')
            
            
        
    