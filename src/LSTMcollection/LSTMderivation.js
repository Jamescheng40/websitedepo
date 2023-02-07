
import React from 'react';
import img from './LSTMappendixA.JPG'
import img2 from './LSTMcellarchitecture.JPG'
import img3 from './gorhdef.JPG'
import img4 from './internalstate.JPG'
import img5 from './LSTMhiddenstate.JPG'
import { MathJaxContext, MathJax } from 'better-react-mathjax';
import * as urlredirect from './redirectutil'
function Home(){
    const config = {
        "HTML-CSS":{ 
            scale: 150
        }
      };

    
    const Ftgeneral = '$$ {F_t = }\\sigma_s{(\\sum_{i=0}^M W_f \\otimes x_t + bias(optional))} = \\sigma_s{( W_f \\odot x_t + bias(optional))}{= \\sigma_s{(d1)} \\; {,where \\; d1=\\sum_{i=0}^M W_f \\otimes x_t \\; is \\; sigmoid \\; and \\; d1=\\sum_{i=0}^M d2}, \\; d2=W_f \\otimes x_t}$$'; 
    const itgeneral = '$$ {i_t = }\\sigma_s{(\\sum_{i=0}^M W_i \\otimes x_t + bias(optional))} = \\sigma_s{( W_i \\odot x_t + bias(optional))} = \\sigma_s{(d6)} \\; {where \\; d6=\\sum_{i=0}^M W_i \\otimes x_t \\; is \\; sigmoid \\; and \\; d6=\\sum_{i=0}^M d7}, \\; d7=W_i \\otimes x_t   $$';
    const Otgeneral = '$$ {O_t = }\\sigma_s{(\\sum_{i=0}^M W_o \\otimes x_t + bias(optional))} = \\sigma_s{( W_o \\odot x_t + bias(optional)) = \\sigma_s{(d13)}} {where \\; d13=\\sum_{i=0}^M W_O \\otimes x_t \\; is \\; sigmoid \\; and \\; d13=\\sum_{i=0}^M d14}, \\; d14=W_o \\otimes x_t  $$';
    const Cdtgeneral = "$$ {C'_t = }\\sigma_t{(\\sum_{i=0}^M W_c \\otimes x_t + bias(optional))} = \\sigma_t{( W_c \\odot x_t + bias(optional))}{= \\sigma_t{(d11)} \\; {,where \\; d11=\\sum_{i=0}^M W_c \\otimes x_t \\; is \\; sigmoid \\; and \\; d11=\\sum_{i=0}^M d12}, \\; d12=W_c \\otimes x_t \\; where \\; \\sigma_t \\; is \\; tanh} $$";
    const Ctgeneral =  "$$ {C_t = }  F_t \\otimes {(C_{t-1})} \\oplus i_t \\otimes C'_t {, \\; where \\; \\otimes \\; is \\; element-wise \\; multiply \\; and \\oplus \\; is \\; element-wise \\; addition  ,where \\; a=F_t \\otimes {(C_{t-1})} \\; and \\; a1= i_t \\otimes C'_t }$$";
    const htgeneral  = "$$ {h_t = }  O_t \\otimes \\sigma_t{(C_t)} = {O_t \\otimes b_t} {, \\; where \\; b_t \\; is \\; \\sigma_t{(C_t)}} $$";
    const Ligeneral = "$$ {L_t =  W_v \\odot h_t =}{\\sum_{i=0}^T W_v \\otimes h_t + bias(optional)}, \\; where \\; L1 =\\sum_{i=0}^M L2,and \\; L2=W_v \\otimes h_t $$"
  //   const Pigeneral = "$$ {P_i = softmax(L_i)}{=\\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}} } $$"
    const Eigeneral = "$$ {E_i=}{ \\frac{1}{n} \\otimes \\sum_{i=0}^T (L_t - Y)^2  }{= \\frac{1}{n} \\otimes \\sum_{i=0}^T g,where \\; g = (L_t - Y)^2}.\\; The \\; Loss \\; function\\; here\\; is\\; the\\; Mean \\; Squred \\; Error $$"
    
    const ColoredLine = ({ color }) => (
        <hr
            style={{
                color: color,
                backgroundColor: color,
                height: 1
            }}
        />
    );
    return (
        <div>
            <h1><strong>&nbsp; &nbsp; LSTM derivation from original Paper(1997) with Python implementation explained in Detail with implementation</strong></h1>

            <p>&nbsp;&nbsp;</p>

            <p><strong><span style={{fontSize:"30px"}}>Background:</span></strong></p>
            <ColoredLine color="#E3E3E3" />
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;This website is for bookkeeping only from paper derivation for future reference and research</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;A lot of websites such as <a href="https://towardsdatascience.com/tutorial-on-lstm-a-computational-perspective-f3417442c2cd">Manu Rastogi</a>,&nbsp;<a href="https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/">Eli Bendersky</a>,&nbsp;and&nbsp;<a href="https://towardsdatascience.com/back-to-basics-deriving-back-propagation-on-simple-rnn-lstm-feat-aidan-gomez-c7f286ba973d" rel="noopener follow">Jae Duk Seo</a> did a great job explaining the details of how LSTM is derived. Few have touched on how to turn them into a script or even explained how LSTM derivation is connected to the implementation itself. So in this article, we will focus on a fully detailed derivation as well as some of the connections it has to the python implementation. In the end, we will try a demonstration on python to see the training of a simple array and the prediction that it has.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;We will keep the original LSTM paper as a reference for&nbsp;<a href="http://www.bioinf.jku.at/publications/older/2604.pdf">Long Short-Term Memory(1997)</a>.</p>

            <p><img src={img3} /></p>

            <p><img src={img} /></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The above image is taken from Appendix A1 of the paper and can be translated into the following:</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Let us also keep an LSTM cell for reference:</p>

            <p><img src={img2} /></p>

            <p><strong><span style={{fontSize:"30px"}}>Forward LSTM:</span></strong></p>
            <ColoredLine color="#E3E3E3" />
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice that the summation&nbsp;<strong>&sum;</strong> here above is simply a<strong> dot product </strong>so we will use an <strong>⨀</strong> to represent it. The&nbsp;<strong>&sigma;s</strong> is sigmoid which is defined above</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1, the forget unit activation between hidden unit and input(as stated above) is</p>
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {Ftgeneral}


                    </MathJax>
                </MathJaxContext>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice that the b is Bias and is optional</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2, the input unit activation between input and activation of input is</p>

            <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {itgeneral}


                    </MathJax>
                </MathJaxContext>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3, the output unit activation between net input and net activation of output is</p>
            <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {Otgeneral}


                    </MathJax>
                </MathJaxContext>
                <p> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;or according to the paper
                </p>
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<img src={img4} /></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice that the <strong>g(net cⱼ)/g(x) and h(x) from the above Appendix</strong>&nbsp;is the special case sigmoid function but we can use <strong>tanh(t)</strong> to represent the same thing since the&nbsp;<strong>g(net cⱼ)</strong>&nbsp;spans the same spectrum(refer to the graph above the value spans from <strong>[-2,2] </strong>or <strong>h(x)</strong> from <strong>[-1,1]</strong>) and <strong>tanh(t)</strong> value spans from<strong>[-1,1]</strong>.&nbsp;<strong>&sigma;t</strong> is represented as <strong>tanh(t)</strong> below.&nbsp;<strong>&otimes;</strong> represent the element-wise multiply</p>

            <p><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</strong>4, the internal state activation between net input and net activation of output is</p>
            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {Cdtgeneral}
                </MathJax>
            </MathJaxContext>
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;5, the internal state (a state that spans across other memory cells) is </p>
            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {Ctgeneral}
                </MathJax>
            </MathJaxContext>
            <p> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;or according to the paper
                </p>
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<img src={img5} /></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 6, the hidden state is computed as</p>
            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {htgeneral}
                </MathJax>
            </MathJaxContext>
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 7, Although the original paper did not state we should use a loss function, we should definitely use a loss function for computing a desired outcome. Here we would choose the <strong>Mean Square Error</strong> to compute a simple array matching. We would do other loss function for other tasks in other posts.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Notice L here represents the Logit function and it is acting as a intermediate hidden layer matrix. It is needed because we can use it to transform the hidden layer to a output layer where we can have our desired output dimension</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {Ligeneral}
                    {Eigeneral}
                </MathJax>
            </MathJaxContext>

            <p><span style={{fontSize:"30px"}}><strong>Backward optimization:</strong></span></p>
            <ColoredLine color="#E3E3E3" />
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; For Backward optimization, the general concept would be Gradient Descent, whose aim is to find the global minimum for each gradient. we can choose either Adam Optimizer or Stochastic Optimizer, they both perform more or less the same. More can refer to the <a href="https://arxiv.org/pdf/1412.6980.pdf">research paper</a>. For our derivation purpose, we choose to use SGD(Stochastic Gradient Descent) to compute the backward gradient.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The loss function we are using is the <strong>Mean Square Error</strong>, the reason we are using the mean square error is because we are computing a simple input array and output array memorization. For more on <strong>Mean Square Error</strong> please check out this <a href="https://en.wikipedia.org/wiki/Mean_squared_error">article</a> that explained what is this function and how this function works.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;For Gradient computation, we should compute the gradient based on&nbsp;<strong><a href={urlredirect.Woderivationpage}>&part;E/&part;Wₒ</a>,&nbsp;<a href={urlredirect.Wfderivationpage}>&part;E/&part;Wf</a>, <a href={urlredirect.Widerivationpage}>&part;E/&part;Wᵢ</a>,&nbsp;<a href={urlredirect.Wctderivationpage}>&part;E/&part;Wc</a>,&nbsp;<a href={urlredirect.Wvderivationpage}>&part;E/&part;Wᵥ</a></strong></p>

            <p><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </strong>Then we should update the weight using&nbsp;<strong>&nbsp; W(any) -=&nbsp;&alpha;</strong>&nbsp;<strong>&otimes; W(any)</strong></p>

            <p><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</strong>For an in-depth derivation of each function please click on the hyperlink.</p>

            <p>&nbsp;</p>

            <p>&nbsp;</p>

        </div>
    )
}
  
export default Home;