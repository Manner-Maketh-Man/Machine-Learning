from tqdm.auto import tqdm
import torch


def train(model,dataloader, optimizer_, scheduler_, device_):
    """
      Train pytorch model on a single pass through the data loader.

      It will use the global variable `model` which is the transformer model
      loaded on `_device` that we want to train on.

      This function is built with reusability in mind: it can be used as is as long
        as the `dataloader` outputs a batch in dictionary format that can be passed
        straight into the model - `model(**batch)`.

      Arguments:

          dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
              Parsed data into batches of tensors.

          optimizer_ (:obj:`transformers.optimization.AdamW`):
              Optimizer used for training.

          scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
              PyTorch scheduler.

          device_ (:obj:`torch.device`):
              Device used to load tensors before feeding to model.

      Returns:

          :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
            Labels, Train Average Loss].
    """

          # Use global variable for model.
    #global model

          # Tracking variables.
    predictions_labels = []
    true_labels = []
          # Total loss for this epoch.
    total_loss = 0

          # Put the model into training mode.
    model.train()

      # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):

            # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

            # move batch to device
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

            # Always clear any previously calculated gradients before performing a
            # backward pass.
        model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this a bert model function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to calculate training accuracy.
        loss, logits = outputs[:2]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
        total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
        loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
        optimizer_.step()

            # Update the learning rate.
        scheduler_.step()

            # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

            # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

      # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

      # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss



def validation(model, dataloader, device_):
    """Validation function to evaluate model performance on a
      separate set of data.

      This function will return the true and predicted labels so we can use later
      to evaluate the model's performance.

      This function is built with reusability in mind: it can be used as is as long
        as the `dataloader` outputs a batch in dictionary format that can be passed
        straight into the model - `model(**batch)`.

      Arguments:

        dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
              Parsed data into batches of tensors.

        device_ (:obj:`torch.device`):
              Device used to load tensors before feeding to model.

      Returns:

        :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
            Labels, Train Average Loss]
    """

    # Use global variable for model.
    #global model

    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss
