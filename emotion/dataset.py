from torch.utils.data import Dataset, DataLoader
import numpy as np


class NewsDataset(Dataset):

    def __init__(self, data, use_tokenizer):
        # Check if path exists.
        """if not os.path.isdir(path):
          # Raise error if path is invalid.
          raise ValueError('Invalid `path` variable! Needs to be a directory')"""
        """self.texts = []
        self.labels = []

        # Since the labels are defined by folders with data we loop 
        # through each label.
        for label in ['ITscience', 'culture','economy','entertainment','health','life','politic','social','sport']:
          sentiment_path = os.path.join(path, label)

          # Get all files from path.
          files_names = os.listdir(sentiment_path)#[:10] # Sample for debugging.
          # Go through each file and read its content.
            for file_name in tqdm(files_names, desc=f'{label} files'):
                file_path = os.path.join(sentiment_path, file_name)

                # Read content.
                content = io.open(file_path, mode='r', encoding='utf-8').read()
                # Fix any unicode issues.
                content = fix_text(content)
                # Save content.
                self.texts.append(content)
                # Save encode labels.
                self.labels.append(label)
        """
        self.texts = np.array(data["context"])
        self.labels = np.array(data["label"])
        # Number of exmaples.
        self.n_examples = len(self.labels)

        return

    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return self.n_examples

    def __getitem__(self, item):
        r"""Given an index return an example from the position.

        Arguments:

          item (:obj:`int`):
              Index position to pick an example to return.

        Returns:
          :obj:`Dict[str, str]`: Dictionary of inputs that contain text and
          asociated labels.

        """

        return {'text': self.texts[item],
                'label': self.labels[item]}

