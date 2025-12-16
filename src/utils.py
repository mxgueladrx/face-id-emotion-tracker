import torch
from collections import Counter
from tqdm import tqdm

def countLabels(ds): # Función para calcular cuantos datos hay de cada clase en un dataset/subset
    if isinstance(ds, torch.utils.data.Subset): # Si es un subset
        return dict(Counter(torch.tensor(ds.dataset.targets)[ds.indices].tolist()))
    else: # Si es un dataset
        return dict(Counter(ds.targets))
    

class EarlyStop():
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.bestValLoss = None
        self.noValImprovementTimes = 0
        self.stop = False

    def checkStop(self, valLoss):
        if self.bestValLoss is None or (valLoss + self.delta) < self.bestValLoss:
            self.bestValLoss = valLoss
            self.noValImprovementTimes = 0
        else:
            self.noValImprovementTimes += 1
            self.stop = self.noValImprovementTimes >= self.patience


def learningLoop(trainDataloader, valDataloader, model, epochs, lossFn, learningRate, optimizer, earlyStop, valFreq, device): # Bucle de entrenamiento para nuestro modelo
    trainLoss = [] # Lista para almacenar el error en cada época de train
    valLoss = [] # Lista para almacenar el error en cada época de validación
    acc = [] # Lista para almacenar nuestro accuracy

    a = 0.0

    opt = optimizer(
        model.parameters(), 
        lr=learningRate
    )

    with tqdm(range(epochs), desc="epoch") as pbar:
        for epoch in pbar:
            model.train()
            trainStepLoss = [] # Lista para almacenar todos los pasos de la época actual
            for xTrainTrue, yTrainTrue in trainDataloader: # Iteramos sobre cada batch de datos del dataloader (cada iteración es un paso)
                xTrainTrue = xTrainTrue.to(device) # Mandamos los datos a cpu o gpu, según la variable
                yTrainTrue = yTrainTrue.to(device)

                yPred = model(xTrainTrue) # Predecimos las salidas de las funciones para la entrada x (forward)
                opt.zero_grad() # Reseteamos los gradientes a 0
                loss = lossFn(yPred, yTrainTrue) # Calculamos el error entre el valor predicho y el real
                loss.backward() # Calculamos los gradientes (backward)
                trainStepLoss.append(loss.clone().detach()) # Almacenamos el error DESVINCULANDOLO del grafo de computación que crea pytorch, para no calcular sus gradientes
                opt.step() # Ajustamos los parametros del modelo usando los gradientes calculados en el backward
            t = torch.tensor(trainStepLoss).mean()
            trainLoss.append(t) # Almacenamos la media de los pasos de cada época
            pbar.set_postfix({"loss" : f"{t}", "acc" : a})

            if epoch % valFreq == 0:
                model.eval()
                valStepLoss = []
                valCorrectAnswers = 0 # Contador para alacenar cuantas respuestas son correctas (el modelo predice bien)
                valSetSize = 0 # Tamaño del conjunto de validación
                with torch.no_grad(): # No calcula los gradientes, lo hacemos para predecir el error y ver como evoluciona la curva de error
                    for xValTrue, yValTrue in valDataloader:
                        xValTrue = xValTrue.to(device)
                        yValTrue = yValTrue.to(device)

                        yValPred = model(xValTrue)
                        loss = lossFn(yValPred, yValTrue)
                        valStepLoss.append(loss)

                        valCorrectAnswers += torch.sum(yValTrue == torch.argmax(yValPred, dim=-1)) # Añadimos al contador cuantos yTrue == yPred
                        valSetSize += yValTrue.numel() # Añadimos el número de elementos
                    v = torch.tensor(valStepLoss).mean()
                    valLoss.append(v)
                    a = valCorrectAnswers / float(valSetSize)
                    acc.append(a) # Calculamos y guardamos el accuracy
                    pbar.set_postfix({"loss" : f"{t}", "acc" : a})
                
                earlyStop.checkStop(v) # Comprobamos si debemos parar
                if earlyStop.stop:
                    print("Early stop")
                    break

    return model, trainLoss, valLoss, acc

