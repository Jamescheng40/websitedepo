import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco,dracula} from 'react-syntax-highlighter/dist/esm/styles/hljs';
function LSTMfullimplementation(){


    const codeString = "import numpy as np\nimport matplotlib.pyplot as plt\nfrom tqdm import tqdm\n";


    // const CodeBlock = ({codestring}) => {
    //         return (
    //         <SyntaxHighlighter language="javascript" style={docco}>
    //             {codeString}
    //         </SyntaxHighlighter>
    //         );
    //     };

    const codeString2 = 'class activation:\n    def sigmoid(self, x):\n        return 1.0 / (1.0 + np.exp(-x))\n    def tanh(self,x):\n        return np.tanh(x)'
    const codeString3 = 'class LSTMmainalgo:\n    def __init__(self, inputarry, outputarry, hidden_dim) -> None:\n        inputsizeArg  = len(inputarry)\n        outputsizeArg = len(outputarry)\n        self.input = inputarry\n        self.output = outputarry\n        self.inputsize = inputsizeArg\n        self.outputsize = outputsizeArg\n        self.hidden_dim = hidden_dim\n\n        #weight initialization\n        self.Wf = np.random.uniform( -1 / np.sqrt(inputsizeArg), 1/  np.sqrt(inputsizeArg), (self.hidden_dim,self.inputsize) )\n        self.Wi = np.random.uniform( -1/  np.sqrt(inputsizeArg), 1/ np.sqrt(inputsizeArg), (self.hidden_dim,self.inputsize)  )\n        self.Wo = np.random.uniform( -1/ np.sqrt(inputsizeArg), 1/ np.sqrt(inputsizeArg), (self.hidden_dim,self.inputsize)   )\n        self.Wc = np.random.uniform( -1/ np.sqrt(inputsizeArg), 1/ np.sqrt(inputsizeArg), (self.hidden_dim,self.inputsize)   )\n        self.Wv = np.random.uniform( -1/ np.sqrt(inputsizeArg), 1/ np.sqrt(inputsizeArg),(self.outputsize,self.hidden_dim)   )\n\n        #storage initialization\n        ct = np.zeros((self.hidden_dim,1))\n        self.ctstorage = [ct]\n        self.oinpstorage = []\n        self.ostorage = []\n        self.astorage = []\n        self.ctdstorage = []\n        self.itinputstorage = []\n        self.a1storage = []\n        self.finputstorage = []\n        self.ctdinpstorage = []\n        self.ftstorage = []\n        self.itstorage = []\n        self.bstorage = []'
    const codeString4 = '    def forward(self, inputarg):\n\n        #forward computation follows the formula defined the webpage\n        for t in range(inputsizeArg):\n            otinp = np.dot(self.Wo, inputarg)\n            itinp = np.dot(self.Wi, inputarg)\n            finp = np.dot(self.Wf, inputarg)\n            ctdinp = np.dot(self.Wc, inputarg)\n            self.ft = activation.sigmoid(self,finp)\n            self.it = activation.sigmoid(self,itinp)\n            self.ot = activation.sigmoid(self, otinp)\n            self.ctd = activation.tanh(self, ctdinp)\n            a = self.ft * self.ctstorage[t]\n            a1 = self.it * self.ctd\n            self.ct = a + a1\n            self.b = activation.tanh(self,self.ct)\n            self.ht = self.ot * self.b\n\n            self.ostorage.append(self.ot)\n            self.itstorage.append(self.it)\n            self.ftstorage.append(self.ft)\n            self.finputstorage.append(finp)\n            self.itinputstorage.append(itinp)\n            self.ctdinpstorage.append(ctdinp)\n            self.ctdstorage.append(self.ctd)\n            self.oinpstorage.append(otinp)\n            self.ctstorage.append(self.ct)\n\n\n\n        self.logit = np.dot(self.Wv, self.ht)\n        self.yhat  = self.softmax(self.logit)'
    const codeString5 = '    def softmax(self, xs):\n    # Applies the Softmax Function to the input array.\n        return np.exp(xs) / sum(np.exp(xs))'
    const codeString6 = '    def backward(self):\n        dEwrtl = self.logit - self.outputencoded\n        dEwrtHwlogitsoftmax = np.dot( np.transpose(dEwrtl), self.Wv)\n        #Wv\n        dEwrtWv = dEwrtl * np.transpose(self.ht)\n        #dwo\n        dEwrtO = dEwrtHwlogitsoftmax * np.transpose( self.b)\n        dEwrtoin = dEwrtO * np.transpose(  activation.sigmoid(self,self.oinpstorage[-1]) * (1 - activation.sigmoid(self,self.oinpstorage[-1])))\n        dEwrtWo = np.transpose(dEwrtoin) * np.transpose(self.input)\n\n        #dwi\n        dEwrtB = dEwrtHwlogitsoftmax * np.transpose(self.ostorage[-1])\n        dEwrtct = dEwrtB * np.transpose(1 - np.tanh(self.ctstorage[-1]) ** 2)\n        dEwrtit = dEwrtct * np.transpose(self.ctdstorage[-1])\n        dEwrtdi = dEwrtit * np.transpose( activation.sigmoid(self,self.itinputstorage[-1]) * (1 - activation.sigmoid(self,self.itinputstorage[-1])))\n        dEwrtWi =  np.transpose(dEwrtdi) * np.transpose(self.input)\n\n        #dwf\n        dEwrtft = dEwrtct * np.transpose(self.ctstorage[-2])\n        dEwrtdf = dEwrtft * np.transpose(activation.sigmoid(self,self.finputstorage[-1]) * (1 - activation.sigmoid(self,self.finputstorage[-1])))\n        dEwrtwf = np.transpose(dEwrtdf) * np.transpose(self.input)\n\n        #dwc\n        dEwrtcdt = dEwrtct * np.transpose(self.itstorage[-1])\n        dEwrtcdtinp = dEwrtcdt * np.transpose(1 - np.tanh(self.ctdinpstorage[-1]) ** 2)\n        dEwrtwc = np.transpose(dEwrtcdtinp) * np.transpose(self.input)\n        dEwrtctprev = dEwrtct\n        for t in reversed(range(inputsizeArg - 1)):\n\n            #dwi\n            dEwrtctprev = dEwrtctprev * np.transpose(self.ftstorage[t + 1])\n            dEwrtit = dEwrtctprev * np.transpose(self.ctdstorage[t])\n            dEwrtdi = dEwrtit * np.transpose(activation.sigmoid(self,self.itinputstorage[t]) * (1 - activation.sigmoid(self,self.itinputstorage[t])) )\n            dEwrtWi += np.transpose(dEwrtdi) * np.transpose(self.input)\n\n            #dwc\n            dEwrtcdt = dEwrtctprev * np.transpose(self.itstorage[t])\n            dEwrtcdtinp = dEwrtcdt * np.transpose(1 - np.tanh(self.ctdinpstorage[t]) ** 2)\n            dEwrtwc += np.transpose(dEwrtcdtinp) * np.transpose(self.input)\n            #dwf\n            dEwrtft = dEwrtctprev * np.transpose(self.ctstorage[t])\n            dEwrtdf = dEwrtft * np.transpose(activation.sigmoid(self,self.itinputstorage[t]) * (1 - activation.sigmoid(self,self.itinputstorage[t])))\n            dEwrtwf += np.transpose(dEwrtdf) * np.transpose(self.input)\n        #Gradient Descent Update\n        self.Wf -= self.lr * dEwrtwf\n        self.Wi -= self.lr * dEwrtWi\n        self.Wo -= self.lr * dEwrtWo\n        self.Wc -= self.lr * dEwrtwc\n        self.Wv -= self.lr * dEwrtWv'
    const codeString7 = '    def train(self, epochs, learning_rate):\n        self.Ovr_loss = []\n        self.lr = learning_rate\n        for epoch in tqdm(range(epochs)):\n            for sample in range(self.input.shape[0]):\n                self.outputencoded = self.output\n                self.forward(self.input)\n                self.backward()\n            self.loss = 0'
    
    const codeString8 = '    def predict(self,x):\n        self.outputs = []\n        self.forward(x)\n        self.outputs.append(self.logit)\n        print(self.outputs)'
    const codeString9 = 'inputarray = np.array([[60],[40],[3]])\noutputarray = np.array([[5],[8],[5],[7],[5],[8],[3],[8]])\nM_hidden_dim = 20\nlstm = LSTMmainalgo(inputarray, outputarray,M_hidden_dim)\nlstm.train(200,1e-2)\nlstm.predict(inputarray)'

    return (
        <div>

<h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <strong>FULL&nbsp;</strong><span style={{fontSize:37.379999999999995}}><strong>Implementation of&nbsp;</strong></span><strong>&nbsp;LSTM using Python with Numpy Only explained in detail</strong></h1>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; This article will be dedicated to the programming aspect of Mathematical derivation.</p>

<p>&nbsp;</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Many implementations for LSTM you see online nowadays can be confusing and few have discovered how the math or derivation behind them works. Today, we will be diving into how to implement the LSTM fully and there will be links to other posts in this website domain explaining how the code is formed. We will be using the Numpy library only as this is an excellent library that can do mathematical operations with reasonable accuracy. We will also provide an implementation in Java for reference. You can find the link&nbsp;<a href="http://a">here</a>.</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Firstly, we need to do the import.</p>

        <SyntaxHighlighter language="python" style={dracula}>
                {codeString}

        </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Secondly, let us code the activation function for different types. One is called sigmoid, the other one is called tanh</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString2}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Thirdly, let us code the initialization weight. There are already some great articles such as <a href="https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/">this</a>&nbsp;that explain how to choose the initial weights for various parameters. We will use the parameters that are proposed by Xavier. In programming, we generally would like to code in modules so we start with class and we will name it&nbsp;LSTMmainalgo. You will probably notice that there are a lot of variables named storage and they are in a hashtable, they are needed because we need to store the computed value for the backpropagation.</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString3}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Next, we will code the forward block for the LSTM function, the LSTM function will follow the general formula as it can be found <a href="http://a">here</a>.</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString4}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The softmax function is next and it is quite simple as we just need to scale the value to a probability</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString5}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The subsequent part will serve as an important optimization step and they are technically called the backpropagation optimization step. A lot of code in the following is followed from the Mathematical derivation from <a href="http://b">here</a></p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString6}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The following function is the function dedicated to training, which would handle the array transformation as well as forward and backward. As stated in the original paper,&nbsp; the method they use is called SGD or Stochastic Gradient Descent. There are other more effective methods such as the Adam optimizer we can use. In order to stick to the original paper, we use the SGD for optimization purposes.</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString7}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The prediction method will require an input array as a parameter and output a predicted result using the forward function</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString8}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The rest of the code will be the input parameter for different applications. Here, for demonstration purposes, we will be using the simple input as an array and the output as another array. The hidden dimension units will be M in quantity. The hidden dimension is explained in the Mathematical Derivation post.</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString9}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The entire python code is following, you can copy and paste as you like and modify the input parameter</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString + "\n" + codeString2+ "\n" + codeString3+ "\n" + codeString4+"\n" + codeString5+ "\n" + codeString6+ "\n" + codeString7+ "\n" + codeString8+"\n" + codeString9}

            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Please check out the following for explaining how the code works and why it works, you can either find the link on the left navigation menu or <a href="http://c">here</a></p>

       </div>
      )

};

export default LSTMfullimplementation;