import torch
from ignite.utils import convert_tensor
from ignite.engine.engine import Engine


def _prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch 
    return (convert_tensor(x, device=device, non_blocking=non_blocking), 
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_supervised_trainer(model, optimizer, loss_fn, device=None, accumulation_steps=1,
                              non_blocking=False, prepare_batch=_prepare_batch):
    if device:
        model.to(device)
    
    def _update(engine, batch):
        model.train()
        if not list(model.features.parameters())[0].requires_grad:
            model.features.eval() # BN in features should be in the eval mode if requires_grad=False
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        
        y_pred = model(x)
        loss = loss_fn(y_pred, y) / accumulation_steps
        loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics, device = None,
                                non_blocking = False, prepare_batch = _prepare_batch,
                                output_transform = lambda x, y, y_pred: (y_pred, y)):
    
    metrics = metrics or {}

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
