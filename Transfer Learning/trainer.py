from torch.autograd import Variable

def train(network, loader, optimizer, criterion, num_iterations):
    for epoch in range(num_iterations):
        running_loss = 0.0
        for i, (images, labels) in enumerate(loader):
            images = Variable(images)
            labels = Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
