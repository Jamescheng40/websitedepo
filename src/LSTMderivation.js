
import React from 'react';
import img from './LSTMappendixA.JPG'
import img2 from './LSTMcellarchitecture.JPG'
import img3 from './gorhdef.JPG'
import img4 from './internalstate.JPG'
import img5 from './LSTMhiddenstate.JPG'
import './main.css';

function Home(){
    return (
        <div>
            <h1><strong>&nbsp; &nbsp; LSTM derivation from original Paper(1997) with Python implementation explained in Detail with implementation</strong></h1>

            <p>&nbsp;&nbsp;</p>

            <p>Background:</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;This website is for bookkeeping only from paper derivation for future reference and research</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;A lot of websites such as <a href="https://towardsdatascience.com/tutorial-on-lstm-a-computational-perspective-f3417442c2cd">Manu Rastogi</a>,&nbsp;<a href="https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/">Eli Bendersky</a>,&nbsp;and&nbsp;<a href="https://towardsdatascience.com/back-to-basics-deriving-back-propagation-on-simple-rnn-lstm-feat-aidan-gomez-c7f286ba973d" rel="noopener follow">Jae Duk Seo</a> did a great job explaining the details of how LSTM is derived. Few have touched on how to turn them into a script or even explained how LSTM derivation is connected to the implementation itself. So in this article, we will focus on a fully detailed derivation as well as some of the connections it has to the python implementation. In the end, we will try a demonstration on python to see the training of a simple array and the prediction that it has.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;We will keep the original LSTM paper as a reference for&nbsp;<a href="http://www.bioinf.jku.at/publications/older/2604.pdf">Long Short-Term Memory(1997)</a>.</p>

            <p><img src={img3} /></p>

            <p><img src={img} /></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The above image is taken from Appendix A1 of the paper and can be translated into the following:</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Let us also keep an LSTM cell for reference:</p>

            <p><img src={img2} /></p>

            <p><strong><span style={{fontSize:"30px"}}>Forward LSTM:</span></strong></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice that the summation&nbsp;<strong>&sum;</strong> here above is simply a<strong> dot product </strong>so we will use an <strong>⨀</strong> to represent it. The&nbsp;<strong>&sigma;s</strong> is sigmoid which is defined above</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1, the forget unit activation between hidden unit and input(as stated above) is&nbsp;<strong> Fₜ =&sigma;s(&nbsp;&sum;Wᵢᵤ yᶸ(t-1)) =&nbsp; &sigma;(Wf ⨀ xₜ + b(optional))</strong></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice that the b is Bias and is optional</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2, the input unit activation between input and activation of input is&nbsp;&nbsp;<strong>Fᵢ =&sigma;s&nbsp;(&sum;Wᵢₙⱼ yᶸ(t-1)) =&nbsp; &sigma;(Wᵢ ⨀ xₜ + b(optional))</strong></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3, the output unit activation between net input and net activation of output is&nbsp;&nbsp;<strong>Fₒ =&nbsp;&sigma;s&nbsp;(&sum;Wₒᵤₜ&nbsp;yᶸ(t-1)) =&nbsp;&sigma;s(Wₒ&nbsp;⨀ xₜ + b(optional))</strong></p>

            <p><img src={img4} /></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice that the <strong>g(net cⱼ)/g(x) and h(x) from the above Appendix</strong>&nbsp;is the special case sigmoid function but we can use <strong>tanh(t)</strong> to represent the same thing since the&nbsp;<strong>g(net cⱼ)</strong>&nbsp;spans the same spectrum(refer to the graph above the value spans from <strong>[-2,2] </strong>or <strong>h(x)</strong> from <strong>[-1,1]</strong>) and <strong>tanh(t)</strong> value spans from<strong>[-1,1]</strong>.&nbsp;<strong>&sigma;t</strong> is represented as <strong>tanh(t)</strong> below.&nbsp;<strong>&otimes;</strong> represent the element-wise multiply</p>

            <p><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</strong>4, the internal state activation between net input and net activation of output is<strong>&nbsp;Fcₜ =&nbsp;&sigma;t(&sum;Wcₜʲ&nbsp;yᶸ(t-1)) =&nbsp;&sigma;(Wc</strong><strong>&nbsp;⨀ xₜ + b(optional))</strong></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;5, the internal state (a state that spans across other memory cells) is <strong>Cₜ =&nbsp;Fₜ&nbsp;&otimes;&nbsp;Fcₜ₋₁&nbsp;+&nbsp;Fᵢ&nbsp;&otimes;&nbsp;&nbsp;Fcₜ</strong></p>

            <p><img src={img5} /></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 6, the hidden state is computed as <strong>hₜ =&nbsp;Fₒ&nbsp;&otimes;&nbsp;&sigma;t(Cₜ)</strong></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 7, As with the same concept as RNN, we should attach a logit function and softmax function after&nbsp;<strong>hₜ </strong>is computed<strong>&nbsp;</strong>multiple times spanning all times.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Notice L here represents the Logit function and softmax takes the form of&nbsp; <strong>eᵃⁱ /&nbsp;&nbsp;&sum;eᵃᵏ</strong></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<strong> L =&nbsp;Wᵥ</strong><strong> ⨀ xₜ + b(optional)</strong></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <strong>Pi = softmax(Li)</strong></p>

            <p>&nbsp;</p>

            <p><span style={{fontSize:"30px"}}><strong>Backward optimization:</strong></span></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; For Backward optimization, the general concept would be Gradient Descent, whose aim is to find the global minimum for each gradient. we can choose either Adam Optimizer or Stochastic Optimizer, they both perform more or less the same. More can refer to the <a href="https://arxiv.org/pdf/1412.6980.pdf">research paper</a>. For our derivation purpose, we choose to use SGD(Stochastic Gradient Descent) to compute the backward gradient.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; We should use the following Cross Entropy Loss function <strong>E = -&nbsp;&sum; Yₜ&nbsp;&otimes; log(Pₜ)</strong>, where&nbsp;<strong>Yₜ</strong> is the true value as entered by the author and the <strong>Pₜ</strong> is the predicted value from the forward pass, to compute our Gradient. There is a nice <a href="https://towardsdatascience.com/what-is-cross-entropy-3bdb04c13616#:~:text=Cross%2Dentropy%20measures%20the%20performance,the%20lower%20the%20cross%2Dentropy.">article</a> that explained what is this function and how this function works.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;For Gradient computation, we should compute the gradient based on&nbsp;<strong><a href="http://localhost:3000/LSTMderivation/dEwrtdWo">&part;E/&part;Wₒ</a>,&nbsp;<a href="http://localhost:3000/LSTMderivation/dEwrtdWf">&part;E/&part;Wf</a>, <a href="http://localhost:3000/LSTMderivation/dEwrtdWi">&part;E/&part;Wᵢ</a>,&nbsp;<a href="http://localhost:3000/LSTMderivation/dEwrtdWc">&part;E/&part;Wc</a>,&nbsp;<a href="http://localhost:3000/LSTMderivation/dEwrtdWv">&part;E/&part;Wᵥ</a></strong></p>

            <p><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </strong>Then we should update the weight using&nbsp;<strong>&nbsp; W(any) -=&nbsp;&alpha;</strong>&nbsp;<strong>&otimes; W(any)</strong></p>

            <p><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</strong>For an in-depth derivation of each function please click on the hyperlink.</p>

            <p>&nbsp;</p>

            <p>&nbsp;</p>

        </div>
    )
}
  
export default Home;