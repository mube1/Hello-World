import torch
class evaluate_model():
    def __init__(self, model, test):
        self.model= model
        self.acc = 0
        self.n_samples = 0
        self.test=test
        #Model here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predicted=None
        
    
    def evaluate(self): 
        self.model.eval()

        with torch.no_grad():
            n_correct = 0
            for images, labels in self.test:
                images = images.reshape(-1, 28*28).to(self.device)
                images=images.reshape(-1,1,28,28).float()
                labels = labels.to(self.device)
                outputs = self.model(images)
                # max returns (value ,index)
                _, self.predicted = torch.max(outputs.data, 1)
                self.n_samples += labels.size(0)
                n_correct += (self.predicted == labels).sum().item()

            self.acc = 100.0 * n_correct / self.n_samples

            return self.acc