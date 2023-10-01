import torch



# Calculate the accuracy
def accuracy(y_pred, y):
    correct = torch.eq(y_pred, y).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc



# Training function
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_func,
               device: torch.device):
    
    train_loss, train_acc = 0, 0
    model.to(device)

    # Start training
    model.train()

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim= 1).argmax(dim= 1)


        # Calculate loss and accuracy
        loss = loss_func(y_logits, y)
        train_loss += loss
        train_acc += accuracy_func(y_pred, y)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Calculate the loss and accuracy per epoch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return (train_loss, train_acc)



#  Testing function
def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               accuracy_func,
               device: torch.device):
    
    test_loss, test_acc = 0, 0

    model.to(device)

    # Start evaluating
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim= 1).argmax(dim= 1)


            # Calculate loss and accuracy
            loss = loss_func(y_logits, y)
            test_loss += loss
            test_acc += accuracy_func(y_pred, y)


        # Calculate the loss and accuracy per epoch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        
        return (test_loss, test_acc)