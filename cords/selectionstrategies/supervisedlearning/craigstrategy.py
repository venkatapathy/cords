import apricot
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data.sampler import SubsetRandomSampler
import math
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import zipfile
from torch.autograd import Variable

vocab=['-']+[chr(ord('a')+i) for i in range(26)]+[chr(ord('A')+i) for i in range(26)]+[chr(ord('0')+i) for i in range(10)]
chrToindex={}
indexTochr={}
cnt=0
for c in vocab:
    chrToindex[c]=cnt
    indexTochr[cnt]=c
    cnt+=1
vocab_size=cnt # uppercase and lowercase English characters and digits(26+26+10=6
sequence_len=28

class CRAIGStrategy(DataSelectionStrategy):
    """
    Implementation of CRAIG Strategy from the paper :footcite:`mirzasoleiman2020coresets` for supervised learning frameworks.

    CRAIG strategy tries to solve the optimization problem given below for convex loss functions:

    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| x^i - x^j \\|

    In the above equation, :math:`\\mathcal{U}` denotes the training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.

    Since, the above optimization problem is not dependent on model parameters, we run the subset selection only once right before the start of the training.

    CRAIG strategy tries to solve the optimization problem given below for non-convex loss functions:

    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| \\nabla_{\\theta} {L_T}^i(\\theta) - \\nabla_{\\theta} {L_T}^j(\\theta) \\|

    In the above equation, :math:`\\mathcal{U}` denotes the training set, :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round,
    and :math:`k` is the budget for the subset. In this case, CRAIG acts an adaptive subset selection strategy that selects a new subset every epoch.

    Both the optimization problems given above are an instance of facility location problems which is a submodular function. Hence, it can be optimally solved using greedy selection methods.

    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    device: str
        The device being utilized - cpu | cuda
    linear_layer: bool
        Apply linear transformation to the data
    if_convex: bool
        If convex or not
    """

    def __init__(self, trainloader, valloader, model, loss,
                 device, linear_layer, if_convex,
                 optimizer='lazy'):
        """
        Constructer method
        """
        super().__init__(trainloader, valloader, model, linear_layer, loss, device)
        self.if_convex = if_convex
        
        self.optimizer = optimizer
        
        self.word_embeddings = self.load_glove_embeddings()
        
    def download_and_unzip_glove(self, url, zip_path, extract_dir):
        """
        Download and unzip GloVe embeddings.

        Parameters
        ----------
        url : str
            URL to download GloVe embeddings.
        zip_path : str
            Path to save the downloaded zip file.
        extract_dir : str
            Directory where to extract the contents of the zip file.
        """
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            print("Downloading GloVe embeddings...")
            response = requests.get(url)
            with open(zip_path, 'wb') as file:
                file.write(response.content)
            print("Download complete. Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            os.remove(zip_path)
            print("Extraction complete.")
        else:
            print("GloVe embeddings already downloaded and extracted.")

    def load_glove_embeddings(self):
        """
        Load GloVe embeddings from a file.

        Parameters
        ----------
        file_path : str
            The path to the GloVe embeddings file.

        Returns
        -------
        dict
            A dictionary where keys are words and values are embeddings.
        """
        glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
        glove_zip_path = "glove.6B.zip"
        glove_dir = "/home/venkat/Projects/Zoho/cords/data/glove"
        self.download_and_unzip_glove(glove_url, glove_zip_path, glove_dir)
        glove_file_path = os.path.join(glove_dir, "glove.6B.300d.txt")

        embeddings_dict = {}
        with open(glove_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        return embeddings_dict

    def distance(self, x, y, exp=2):
        """
        Compute the distance.

        Parameters
        ----------
        x: Tensor
            First input tensor
        y: Tensor
            Second input tensor
        exp: float, optional
            The exponent value (default: 2)

        Returns
        ----------
        dist: Tensor
            Output tensor
        """

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, exp).sum(2)
        # dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model_params, idxs):
        """
        Compute the score of the indices.

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        idxs: list
            The indices
        """
        # Load a subset of data based on the provided indices
        trainset = self.trainloader.sampler.data_source
        subset_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
                                                    sampler=SubsetRandomSampler(idxs),
                                                    pin_memory=True)
          # Load model parameters
        self.model.load_state_dict(model_params)
         # Initialize counters and lists
        self.N = 0 # Number of samples processed
        g_is = [] # List to store gradients or features

        # Compute scores based on whether the model is convex or not
        if self.if_convex:
            for batch_idx, (inputs, targets) in enumerate(subset_loader):
                # For convex models, process inputs as is
                inputs, targets = inputs, targets
            
                self.N += inputs.size()[0]
                # Flatten inputs and append to list
                g_is.append(inputs.view(inputs.size()[0], -1))
        else:
            # For non-convex models, process using model's embeddings
            embDim = self.model.get_embedding_dim()
            for batch_idx, (inputs, targets) in enumerate(subset_loader):
                inputs = inputs.to(self.device)
                out_size=Variable(torch.IntTensor([sequence_len] * len(targets)))
                y_size=Variable(torch.IntTensor([len(l) for l in targets]))
                conc_label=''.join(targets)          
                y=[chrToindex[c] for c in conc_label]
                y_var=Variable(torch.IntTensor(y))
                #y_var=y_var.to(self.device)
                
                self.N += inputs.size()[0]
               
                # Get model outputs and calculate loss
                out, l1 = self.model(inputs, freeze=True, last=True)
                loss = self.loss(out,y_var,out_size, y_size).sum()
                l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    # If linear layer is used, expand gradients and concatenate
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    g_is.append(torch.cat((l0_grads, l1_grads), dim=1))
                else:
                    # If no linear layer, use the gradients directly
                    # Flatten the gradients and append to g_is
                    # Assuming l0_grads has the shape [28, 20, 63]
                    l0_grads_transposed = l0_grads.transpose(0, 1)  # Transpose to [20, 28, 63]
                    l0_grads_flattened = l0_grads_transposed.reshape(20, -1)  # Flatten to [20, 28*63]

                    g_is.append(l0_grads_flattened)
        # Initialize the distance matrix
        self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
        first_i = True
        
        # Compute pairwise distances between all gradients/features
        for i, g_i in enumerate(g_is, 0):
            # Calculate the start and end indices for the i-th batch
            start_i = sum(g.size(0) for g in g_is[:i])
            end_i = start_i + g_i.size(0)

            for j, g_j in enumerate(g_is, 0):
                # Calculate the start and end indices for the j-th batch
                start_j = sum(g.size(0) for g in g_is[:j])
                end_j = start_j + g_j.size(0)

                # Update distance matrix with calculated distances
                self.dist_mat[start_i:end_i, start_j:end_j] = self.distance(g_i, g_j).cpu()

        # Normalize the distance matrix
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()

    def compute_gamma(self, idxs):
        """
        Compute the gamma values for the indices.

        Parameters
        ----------
        idxs: list
            The indices

        Returns
        ----------
        gamma: list
            Gradient values of the input indices
        """

        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)
        for i in range(rep.shape[1]):
            gamma[rep[0, i]] += 1
        return gamma

    def get_similarity_kernel(self):
        """
        Obtain the similarity kernel for OCR targets.

        Returns
        ----------
        kernel: ndarray
            Array of kernel values based on word similarity.
        """
        embeddings = []
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # Here, 'targets' are expected to be words or sequences of characters.
            # Convert these words into embeddings. You can use a pre-trained embedding model like Word2Vec, GloVe, etc.
            word_embeddings = self.embed_words(targets)  # This is a placeholder function.
            embeddings.extend(word_embeddings)
        
        embeddings = np.array(embeddings)
        # Using cosine similarity to calculate the similarity between word embeddings.
        kernel = cosine_similarity(embeddings)
        
        return kernel

    def select(self, budget, model_params):
        """
        Data selection method using different submodular optimization
        functions.

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        optimizer: str
            The optimization approach for data selection. Must be one of
            'random', 'modular', 'naive', 'lazy', 'approximate-lazy', 'two-stage',
            'stochastic', 'sample', 'greedi', 'bidirectional'

        Returns
        ----------
        total_greedy_list: list
            List containing indices of the best datapoints
        gammas: list
            List containing gradients of datapoints present in greedySet
        """
        # Load all labels from the training data
        # for batch_idx, (inputs, targets) in enumerate(self.trainloader):
        #     if batch_idx == 0:
        #         labels = targets
        #     else:
        #         tmp_target_i = targets
        #         labels = torch.cat((labels, tmp_target_i), dim=0)
        # per_class_bud = int(budget / self.num_classes)
        # Initialize lists for storing selected indices and their corresponding scores
        total_greedy_list = []
        gammas = []
        idxs = torch.arange(0, self.N_trn).long()
        N = len(idxs)
        self.compute_score(model_params, idxs)
        row = idxs.repeat_interleave(N)
        col = idxs.repeat(N)
        data = self.dist_mat.flatten()
        sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
        self.dist_mat = sparse_simmat
        fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                            n_samples=budget,
                                                                            optimizer=self.optimizer)
        sim_sub = fl.fit_transform(sparse_simmat)
        total_greedy_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
        gammas = self.compute_gamma(total_greedy_list)
        # Return the final list of selected indices and their scores
        return total_greedy_list, gammas
