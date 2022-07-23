import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


X = np.array([[1, 3, 0], [2, 4, 1], [3, 2, 1], [2, 1, 3]])

def create_graph(X, idxTrain, knn):
    zeroTolerance = 1e-9
    N = X.shape[1]

    XTrain = np.transpose(X[idxTrain, :])

    # Calculating correlation matrix
    binaryTemplate = (XTrain > 0).astype(XTrain.dtype)
    sumMatrix = XTrain.dot(binaryTemplate.T)
    countMatrix = binaryTemplate.dot(binaryTemplate.T)
    countMatrix[countMatrix == 0] = 1
    avgMatrix = sumMatrix / countMatrix
    sqSumMatrix = (XTrain ** 2).dot(binaryTemplate.T)
    correlationMatrix = sqSumMatrix / countMatrix - avgMatrix ** 2

    # Normalizing by diagonal weights
    sqrtDiagonal = np.sqrt(np.diag(correlationMatrix))
    nonzeroSqrtDiagonalIndex = (sqrtDiagonal > zeroTolerance).astype(sqrtDiagonal.dtype)
    sqrtDiagonal[sqrtDiagonal < zeroTolerance] = 1.
    invSqrtDiagonal = 1/sqrtDiagonal
    invSqrtDiagonal = invSqrtDiagonal * nonzeroSqrtDiagonalIndex
    normalizationMatrix = np.diag(invSqrtDiagonal)

    # Zero-ing the diagonal
    normalizedMatrix = normalizationMatrix.dot(
                            correlationMatrix.dot(normalizationMatrix)) \
                            - np.eye(correlationMatrix.shape[0])

    # Keeping only edges with weights above the zero tolerance
    normalizedMatrix[np.abs(normalizedMatrix) < zeroTolerance] = 0.
    W = normalizedMatrix

    # Sparsifying the graph
    WSorted = np.sort(W, axis=1)
    threshold = WSorted[:, -knn].squeeze()
    thresholdMatrix = (np.tile(threshold, (N, 1))).transpose()
    W[W < thresholdMatrix] = 0

    # Normalizing by eigenvalue with largest magnitude
    E, V = np.linalg.eig(W)
    W = W/np.max(np.abs(E))

    return W

# Creating and sparsifying the graph

nTotal = X.shape[0]
permutation = np.random.permutation(nTotal)
nTrain = int(np.ceil(0.5*nTotal))
idxTrain = permutation[0:nTrain]
nTest = nTotal-nTrain
idxTest = permutation[nTrain:nTotal]

W = torch.Tensor(create_graph(X=X, idxTrain=idxTrain, knn=3))
print(W)

def split_data(X, idxTrain, idxTest, idxMovie):
    N = X.shape[1]

    xTrain = X[idxTrain, :]
    idx = np.argwhere(xTrain[:, idxMovie] > 0).squeeze()
    xTrain = xTrain[idx, :]
    yTrain = np.zeros(xTrain.shape)
    yTrain[:, idxMovie] = xTrain[:, idxMovie]
    xTrain[:, idxMovie] = 0

    xTrain = torch.tensor(xTrain).float()
    xTrain = xTrain.reshape([-1, 1, N])
    yTrain = torch.tensor(yTrain).float()
    yTrain = yTrain.reshape([-1, 1, N])

    xTest = X[idxTest, :]
    idx = np.argwhere(xTest[:, idxMovie] > 0).squeeze()
    xTest = xTest[idx, :]
    yTest = np.zeros(xTest.shape)
    yTest[:, idxMovie] = xTest[:, idxMovie]
    xTest[:, idxMovie] = 0

    xTest = torch.tensor(xTest).float()
    xTest = xTest.reshape([-1, 1, N])
    yTest = torch.tensor(yTest).float()
    yTest = yTest.reshape([-1, 1, N])

    return xTrain, yTrain, xTest, yTest


xTrain, yTrain, xTest, yTest = split_data(X, idxTrain, idxTest, 1)
nTrain = xTrain.shape[0]
nTest = xTest.shape[0]


def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """
    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.
    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.

    # If we can't recognize the type, we just make everything numpy.

    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype

    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype=dataType)

    # This only converts between numpy and torch. Any other thing is ignored
    return x


def permIdentity(S):
    """
    permIdentity: determines the identity permnutation
    Input:
        S (np.array): matrix
    Output:
        permS (np.array): matrix permuted (since, there's no permutation, it's
              the same input matrix)
        order (list): list of indices to make S become permS.
    """
    assert len(S.shape) == 2 or len(S.shape) == 3
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        S = S.reshape([1, S.shape[0], S.shape[1]])
        scalarWeights = True
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
    # Number of nodes
    N = S.shape[1]
    # Identity order
    order = np.arange(N)
    # If the original GSO assumed scalar weights, get rid of the extra dimension
    if scalarWeights:
        S = S.reshape([N, N])

    return S, order.tolist()


class NoPool(nn.Module):
    """
    This is a pooling layer that actually does no pooling. It has the same input
    structure and methods of MaxPoolLocal() for consistency. Basically, this
    allows us to change from pooling to no pooling without necessarily creating
    a new architecture.

    In any case, we're pretty sure this function should never ship, and pooling
    can be avoided directly when defining the architecture.
    """

    def __init__(self, nInputNodes, nOutputNodes, nHops):

        super().__init__()
        self.nInputNodes = nInputNodes
        self.nOutputNodes = nOutputNodes
        self.nHops = nHops
        self.neighborhood = None

    def addGSO(self, GSO):
        # This is necessary to keep the form of the other pooling strategies
        # within the SelectionGNN framework. But we do not care about any GSO.
        pass

    def forward(self, x):
        # x should be of shape batchSize x dimNodeSignals x nInputNodes
        assert x.shape[2] == self.nInputNodes
        # Check that there are at least the same number of nodes that
        # we will keep (otherwise, it would be unpooling, instead of
        # pooling)
        assert x.shape[2] >= self.nOutputNodes
        # And do not do anything
        return x


class LocalGNN(nn.Module):
    """
    LocalGNN: implement the selection GNN architecture where all operations are
        implemented locally, i.e. by means of neighboring exchanges only. More
        specifically, it has graph convolutional layers, but the readout layer,
        instead of being an MLP for the entire graph signal, it is a linear
        combination of the features at each node.
        >> Obs.: This precludes the use of clustering as a pooling operation,
            since clustering is not local (it changes the given graph).
    Initialization:
        LocalGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                 nonlinearity, # Nonlinearity
                 nSelectedNodes, poolingFunction, poolingSize, # Pooling
                 dimReadout, # Local readout layer
                 GSO, order = None # Structure)
        Input:
            /** Graph convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nFilterTaps (list of int): number of filter taps on each layer
                (i.e. nFilterTaps-1 is the extent of neighborhoods that are
                 reached, for example K=2 is info from the 1-hop neighbors)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.

            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations

            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer

            /** Readout layers **/
            dimReadout (list of int): number of output hidden units of a
                sequence of fully connected layers applied locally at each node
                (i.e. no exchange of information involved).

            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
        Output:
            nn.Module with a Local GNN architecture with the above specified
            characteristics.
    Forward call:
        LocalGNN(x)
        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes
        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimReadout[-1] x nSelectedNodes[-1]

    Other methods:

        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.

        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimReadout[-1], as well as the output
        of all the GNN layers (i.e. before the readout layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.

        y = .singleNodeForward(x, nodes): outputs the value of the last layer
        at a single node. x is the usual input of shape batchSize x dimFeatures
        x numberNodes. nodes is either a single node (int) or a collection of
        nodes (list or numpy.array) of length batchSize, where for each element
        in the batch, we get the output at the single specified node. The
        output y is of shape batchSize x dimReadout[-1].
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimReadout,
                 # Structure
                 GSO,
                 order=None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.alpha = poolingSize
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimReadout = dimReadout
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(GraphFilter(self.F[l], self.F[l+1], self.K[l], self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimReadout) > 0: # Maybe we don't want to readout anything
            # The first layer has to connect whatever was left of the graph 
            # filtering stage to create the number of features required by
            # the readout layer
            fc.append(nn.Linear(self.F[-1], dimReadout[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimReadout)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimReadout[l], dimReadout[l+1],
                                    bias = self.bias))
        # And we're done
        self.Readout = nn.Sequential(*fc)
        # so we finally have the architecture.

    def splitForward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        yGFL = self.GFL(x)
        # Change the order, for the readout
        y = yGFL.permute(0, 2, 1) # B x N[-1] x F[-1]
        # And, feed it into the Readout layer
        y = self.Readout(y) # B x N[-1] x dimReadout[-1]
        # Reshape and return
        return y.permute(0, 2, 1), yGFL
        # B x dimReadout[-1] x N[-1], B x dimFeatures[-1] x N[-1]

    def forward(self, x):
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S)
            self.GFL[3*l+2].addGSO(self.S)

import math


def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.
    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.
    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.
    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}
    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N
    print("LSIGF shapes:")
    print(f"h:{h.shape}")
    print(f"S:{S.shape}")
    print(f"x:{x.shape}")
    print(f"b:{b.shape}")

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions

    print("LSIGF first step: reshape x S and create z")
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1) # This is for k = 0

    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1, K):
        x = torch.matmul(x, S) # B x E x G x N
        xS = x.reshape([B, E, 1, G, N]) # B x E x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x E x k x G x N
        print(f"z :{z.shape}")
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    print(z.shape)
    print(h.shape)
    y = torch.matmul(z.permute(0, 4, 1, 2, 3).reshape([B, N, E*K*G]),
                     h.reshape([F, E*K*G]).permute(1, 0)).permute(0, 2, 1)
    print(f"y shape: {y.shape}")
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y


class GraphFilter(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter
    Initialization:
        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)
        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering
        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).
        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features
    Add graph shift operator:
        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).
        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes
    Forward call:
        y = GraphFilter(x)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes
        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        # Compute the filter output
        u = LSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        return u


def movieMSELoss(yHat, y, idxMovie):
    mse = nn.MSELoss()
    return mse(yHat[:,:,idxMovie].reshape([-1,1]),y[:,:,idxMovie].reshape([-1,1]))


GNN1Ly = LocalGNN([1, 64], [5], True, nn.ReLU, [3], NoPool, [1], [1], W)
nEpochs = 1
batchSize = 10
learningRate = 0.005
optimizers = optim.Adam(GNN1Ly.parameters(), lr=learningRate)
GNN1Ly.zero_grad()
print(xTrain)
yHatTrainBatch = GNN1Ly(xTrain.float())
lossValueTrain = movieMSELoss(yHatTrainBatch, yTrain, 1)
print(lossValueTrain)
# Compute gradients
lossValueTrain.backward()

# Optimize
optimizers.step()
