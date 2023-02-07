import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco,dracula} from 'react-syntax-highlighter/dist/esm/styles/hljs';
import * as urlredirect from './redirectutil'

function LSTMbackward(){
    const config = {
        "HTML-CSS":{ 
            scale: 150
        }
      };

const codeString6 = '    def backward(self):\n        dEwrtl = self.logit - self.outputencoded\n        #Wv\n        dEwrtWv = dEwrtl * np.transpose(self.ht)\n\n        dEwrtdht = np.dot( np.transpose(dEwrtl), self.Wv)\n        #dwo\n        dEwrtO = dEwrtdht * np.transpose( self.b)\n        dEwrtoin = dEwrtO * np.transpose(  activation.sigmoid(self,self.oinpstorage[-1]) * (1 - activation.sigmoid(self,self.oinpstorage[-1])))\n        dEwrtWo = np.transpose(dEwrtoin) * np.transpose(self.input)\n\n        #dwi\n        dEwrtB = dEwrtdht * np.transpose(self.ostorage[-1])\n        dEwrtct = dEwrtB * np.transpose(1 - np.tanh(self.ctstorage[-1]) ** 2)\n        dEwrtit = dEwrtct * np.transpose(self.ctdstorage[-1])\n        dEwrtdi = dEwrtit * np.transpose( activation.sigmoid(self,self.itinputstorage[-1]) * (1 - activation.sigmoid(self,self.itinputstorage[-1])))\n        dEwrtWi =  np.transpose(dEwrtdi) * np.transpose(self.input)\n\n        #dwf\n        dEwrtft = dEwrtct * np.transpose(self.ctstorage[-2])\n        dEwrtdf = dEwrtft * np.transpose(activation.sigmoid(self,self.finputstorage[-1]) * (1 - activation.sigmoid(self,self.finputstorage[-1])))\n        dEwrtwf = np.transpose(dEwrtdf) * np.transpose(self.input)\n\n        #dwc\n        dEwrtcdt = dEwrtct * np.transpose(self.itstorage[-1])\n        dEwrtcdtinp = dEwrtcdt * np.transpose(1 - np.tanh(self.ctdinpstorage[-1]) ** 2)\n        dEwrtwc = np.transpose(dEwrtcdtinp) * np.transpose(self.input)\n        dEwrtctprev = dEwrtct\n        for t in reversed(range(self.inputsize - 1)):\n\n            #dwi\n            dEwrtctprev = dEwrtctprev * np.transpose(self.ftstorage[t + 1])\n            dEwrtit = dEwrtctprev * np.transpose(self.ctdstorage[t])\n            dEwrtdi = dEwrtit * np.transpose(activation.sigmoid(self,self.itinputstorage[t]) * (1 - activation.sigmoid(self,self.itinputstorage[t])) )\n            dEwrtWi += np.transpose(dEwrtdi) * np.transpose(self.input)\n\n            #dwc\n            dEwrtcdt = dEwrtctprev * np.transpose(self.itstorage[t])\n            dEwrtcdtinp = dEwrtcdt * np.transpose(1 - np.tanh(self.ctdinpstorage[t]) ** 2)\n            dEwrtwc += np.transpose(dEwrtcdtinp) * np.transpose(self.input)\n            #dwf\n            dEwrtft = dEwrtctprev * np.transpose(self.ctstorage[t])\n            dEwrtdf = dEwrtft * np.transpose(activation.sigmoid(self,self.itinputstorage[t]) * (1 - activation.sigmoid(self,self.itinputstorage[t])))\n            dEwrtwf += np.transpose(dEwrtdf) * np.transpose(self.input)\n        #Gradient Descent Update\n        self.Wf -= self.lr * dEwrtwf\n        self.Wi -= self.lr * dEwrtWi\n        self.Wo -= self.lr * dEwrtWo\n        self.Wc -= self.lr * dEwrtwc\n        self.Wv -= self.lr * dEwrtWv'


const codeString1 = 'dEwrtl = self.logit - self.outputencoded'

const dEoverdwv = "$$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1}  $$"
         
const Eigeneral = "$$ {E_i=}{ \\frac{1}{n} \\otimes \\sum_{i=0}^T (L_t - Y)^2  }{= \\frac{1}{n} \\otimes \\sum_{i=0}^T g,where \\; g = (L_t - Y)^2}.\\; The \\; Loss \\; function\\; here\\; is\\; the\\; Mean \\; Squred \\; Error $$"


const dEoverdG_backward  = "$$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} =  \\begin{bmatrix} \\frac{1}{n} & \\frac{1}{n} & \\cdots & \\frac{1}{n}  \\end{bmatrix}  \\odot \\begin{bmatrix} \\frac{\\partial (L_1 - Y_1)^2}{\\partial L_1} & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_1 - Y_1)^2}{\\partial L_T}} \\\\ \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_1}} & \\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_T}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}} & \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_2}} & \\cdots & \\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}  \\end{bmatrix} = \\begin{bmatrix} \\frac{1}{n} & \\frac{1}{n} & \\cdots & \\frac{1}{n}  \\end{bmatrix} \\odot \\begin{bmatrix} 2 \\times (L_1 - Y_1) & 0 & \\cdots & 0 \\\\ 0 & 2 \\times (L_2 - Y_2) & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 2 \\times (L_T - Y_T)  \\end{bmatrix}  $$" 
         
const dEoverdG_backward_cont3 = "$$ = \\begin{bmatrix} (\\frac{1}{n} \\times 2 \\times (L_1-Y_1)  ) &  ( \\frac{1}{n} \\times 2 \\times (L_2-Y_2)) & \\cdots &  (\\frac{1}{n} \\times 2 \\times (L_T-Y_T) )   \\end{bmatrix}  $$"
         
const dEoverdG_backward_cont5 = "$$ = \\begin{bmatrix} (\\frac{2}{n}  \\times (L_1-Y_1)  ) &  ( \\frac{2}{n} \\times (L_2-Y_2)) & \\cdots &  (\\frac{2}{n} \\times (L_T-Y_T) )   \\end{bmatrix}  $$"
        

const powerrule = "$$ \\frac{\\partial x^n}{\\partial x} = n x^{n-1} $$"

const Wvprecode = "dEwrtWv = dEwrtl * np.transpose(self.ht)"
const Wvlater = "$$ \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial W_v} = \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & h_1 & h_2 & \\cdots & h_M \\end{bmatrix} \\ni T \\times MT $$"
const dEoverdWv = "$$ \\frac{\\partial E}{\\partial W_v} = \\begin{bmatrix} h_1 \\times (L_1 - Y_1) & h_2 \\times (L_1 - Y_1) & \\cdots & h_M \\times (L_1 - Y_1) & h_1 \\times (L_2 - Y_2) & h_2 \\times (L_2 - Y_2) & \\cdots & h_M \\times (L_2 - Y_2) & \\cdots & \\cdots & h_1 \\times (L_3 - Y_3) & h_2 \\times (L_3 - Y_3) \\cdots h_M \\times (L_T - Y_T) \\end{bmatrix} \\ni 1 \\times MT $$"
const dEoverdWvpre = "$$ \\frac{\\partial E}{\\partial W_v} = \\begin{bmatrix} (\\frac{1}{n} \\times 2 \\times (L_1-Y_1)  ) &  ( \\frac{1}{n} \\times 2 \\times (L_2-Y_2)) & \\cdots &  (\\frac{1}{n} \\times 2 \\times (L_T-Y_T) )   \\end{bmatrix} \\odot \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & h_1 & h_2 & \\cdots & h_M \\end{bmatrix}  $$"
const dEoverdWvfinal = "$$ \\frac{\\partial E}{\\partial W_v} = \\begin{bmatrix} h_1 \\times (L_1 - Y_1) & h_2 \\times (L_1 - Y_1) & \\cdots & h_M \\times (L_1 - Y_1) \\\\ h_1 \\times (L_2 - Y_2) & h_2 \\times (L_2 - Y_2) & \\cdots & h_M \\times (L_2 - Y_2) \\\\  h_1 \\times (L_3 - Y_3) & h_2 \\times (L_3 - Y_3) & \\cdots & h_M \\times (L_T - Y_T) \\end{bmatrix} \\ni T \\times M $$"

const transposedemonstration = "import numpy\na = np.array([[1],[2],[3],[4],[5],[6],[7]])        # a is of dimension 7 x 1\nb = np.array([6,7,8,9,10])        # b is of dimension 1 x 5\na*b    # result is a 7 x 5 matrix"

const woprecode = "dEwrtdht = np.dot( np.transpose(dEwrtl), self.Wv)\n#dwo\ndEwrtO = dEwrtdht * np.transpose( self.b)\ndEwrtoin = dEwrtO * np.transpose(  activation.sigmoid(self,self.oinpstorage[-1]) * (1 - activation.sigmoid(self,self.oinpstorage[-1])))\ndEwrtWo = np.transpose(dEwrtoin) * np.transpose(self.input)"

const dEoverdWo = "$$ \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot  ( \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\underline{\\textcolor{blue}{\\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_{t-1}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_{t-1}}{\\partial a_{t-1}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot( \\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus { \\frac{\\partial h_{t-3}}{\\partial O_{t-3}}  \\odot \\frac{\\partial O_{t-3}}{\\partial d13_{t-3}} \\odot \\frac{\\partial d13_{t-3}}{\\partial d14_{t-3}} \\odot \\frac{\\partial d14_{t-3}}{\\partial W_o} } ) \\oplus \\frac{\\partial h_{t-2}}{\\partial O_{t-2}}  \\odot \\frac{\\partial O_{t-2}}{\\partial d13_{t-2}} \\odot \\frac{\\partial d13_{t-2}}{\\partial d14_{t-2}} \\odot \\frac{\\partial d14_{t-2}}{\\partial W_o} )\\oplus \\odot { \\frac{\\partial h_{t-1}}{\\partial O_{t-1}}  \\odot \\frac{\\partial O_{t-1}}{\\partial d13_{t-1}} \\odot \\frac{\\partial d13_{t-1}}{\\partial d14_{t-1}} \\odot \\frac{\\partial d14_{t-1}}{\\partial W_o} } ) \\oplus { \\frac{\\partial h_{t}}{\\partial O_{t}}  \\odot \\frac{\\partial O_{t}}{\\partial d13_{t}} \\odot \\frac{\\partial d13_{t}}{\\partial d14_{t}} \\odot \\frac{\\partial d14_{t}}{\\partial W_o} } ) $$" 


const dEoverdWosimplifiedcancledcont = "$$ = \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot { \\frac{\\partial h_t}{\\partial O_t}  \\odot \\frac{\\partial O_t}{\\partial d13} \\odot \\frac{\\partial d13}{\\partial d14} \\odot \\frac{\\partial d14}{\\partial W_o} } $$"


const dEoverdht = "$$ \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} = \\begin{bmatrix} (\\frac{2}{n}  \\times (L_1-Y_1)  ) &  ( \\frac{2}{n} \\times (L_2-Y_2)) & \\cdots &  (\\frac{2}{n} \\times (L_T-Y_T) )   \\end{bmatrix} \\odot \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix}  $$"

const dLoverdht = "$$ \\frac{\\partial Li}{\\partial h_t} = \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix} \\ni T \\times M $$"

const dhtoverWo = "$$ \\frac{\\partial h_{t}}{\\partial O_{t}}  \\odot \\frac{\\partial O_{t}}{\\partial d13_{t}} \\odot \\frac{\\partial d13_{t}}{\\partial d14_{t}} \\odot \\frac{\\partial d14_{t}}{\\partial W_o} =   diag(\\sigma_t(C_t)) \\odot diag(sigmoid(d13_t) \\times (1 - sigmoid(d13_t))) \\odot \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix}  $$"

const dhtoverWopre = "$$ \\begin{bmatrix} x_1 \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) & x_2 \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) & \\cdots & x_N \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) & x_1 \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2)) & x_2 \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2)) & \\cdots & x_N \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2)) & \\cdots & \\cdots & x_1 \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) & x_2 \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) & \\cdots \\ x_N \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) \\end{bmatrix} $$"

const dhtoverWocont = "$$= \\begin{bmatrix} x_1 \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1))  & x_2 \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) & \\cdots & x_N \\times \\sigma_t(C_1) \\times (sigmoid(d13_1) \\times (1 - sigmoid(d13_1)) \\\\ x_1 \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2))  & x_2 \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2))  & \\cdots & x_N \\times \\sigma_t(C_2) \\times (sigmoid(d13_2) \\times (1 - sigmoid(d13_2))  \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\\\ x_1 \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) & x_2 \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) & \\cdots & x_N \\times \\sigma_t(C_M) \\times (sigmoid(d13_M) \\times (1 - sigmoid(d13_M)) \\end{bmatrix} $$"

const wiprecode = "#dwi\ndEwrtB = dEwrtdht * np.transpose(self.ostorage[-1])\ndEwrtct = dEwrtB * np.transpose(1 - np.tanh(self.ctstorage[-1]) ** 2)\ndEwrtit = dEwrtct * np.transpose(self.ctdstorage[-1])\ndEwrtdi = dEwrtit * np.transpose( activation.sigmoid(self,self.itinputstorage[-1]) * (1 - activation.sigmoid(self,self.itinputstorage[-1])))\ndEwrtWi =  np.transpose(dEwrtdi) * np.transpose(self.input)"

const ddoverd1timesd21overdWocombine1 = "$$ \\frac{\\partial d13}{\\partial W_o} = \\frac{\\partial d13}{\\partial d14} \\odot \\frac{\\partial d14}{\\partial W_o} = \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} \\ni M \\times MN  $$"
const tequal3computehighlighted = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\underline{ \\textcolor{blue}{\\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot(\\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus \\frac{\\partial C_{t-3}}{\\partial a1_{t-3}} \\odot \\frac{\\partial a1_{t-3}}{\\partial i_{t-3}} \\odot \\frac{\\partial i_{t-3}}{\\partial d6_{t-3}} \\odot \\frac{\\partial d6_{t-3}}{\\partial d7_{t-3}} \\odot \\frac{\\partial d7_{t-3}}{\\partial W_{i_{t-3}}}) \\oplus \\frac{\\partial C_{t-2}}{\\partial a1_{t-2}} \\odot \\frac{\\partial a1_{t-2}}{\\partial i_{t-2}} \\odot \\frac{\\partial i_{t-2}}{\\partial d6_{t-2}} \\odot \\frac{\\partial d6_{t-2}}{\\partial d7_{t-2}} \\odot \\frac{\\partial d7_{t-2}}{\\partial W_{i_{t-2}}} )\\oplus \\frac{\\partial C_{t-1}}{\\partial a1_{t-1}} \\odot \\frac{\\partial a1_{t-1}}{\\partial i_{t-1}} \\odot \\frac{\\partial i_{t-1}}{\\partial d6_{t-1}} \\odot \\frac{\\partial d6_{t-1}}{\\partial d7_{t-1}} \\odot \\frac{\\partial d7_{t-1}}{\\partial W_{i_{t-1}}} ) \\oplus  \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot \\frac{\\partial d7}{\\partial W_i} ) $$"

const dEoverdCt = "$$ \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} = \\begin{bmatrix} (\\frac{2}{n}  \\times (L_1-Y_1)  ) &  ( \\frac{2}{n} \\times (L_2-Y_2)) & \\cdots &  (\\frac{2}{n} \\times (L_T-Y_T) )   \\end{bmatrix} \\odot \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix} \\odot diag(O_t) \\odot (1 - tanh^2(C_t)) $$"

const dCtoverdWi = "$$ \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot \\frac{\\partial d7}{\\partial W_i} = 1 \\odot diag(C'_t) \\odot (sigmoid(d5_t) \\times (1 - sigmoid(d5_t))) \\odot \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} $$"
const dCtoverdWicont = "$$ \\begin{bmatrix} x_1 \\times C'_1 \\times (sigmoid(d5_1) \\times (1 - sigmoid(d5_1))  & x_2 \\times C'_1 \\times (sigmoid(d5_1) \\times (1 - sigmoid(d5_1)) & \\cdots & x_N \\times C'_1 \\times (sigmoid(d5_1) \\times (1 - sigmoid(d5_1)) \\\\ x_1 \\times C'_2 \\times (sigmoid(d5_2) \\times (1 - sigmoid(d5_2))  & x_2 \\times C'_2 \\times (sigmoid(d5_2) \\times (1 - sigmoid(d5_2))  & \\cdots & x_N \\times C'_2 \\times (sigmoid(d5_2) \\times (1 - sigmoid(d5_2))  \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\\\ x_1 \\times C'_M \\times (sigmoid(d5_M) \\times (1 - sigmoid(d5_M)) & x_2 \\times C'_M \\times (sigmoid(d5_M) \\times (1 - sigmoid(d5_M)) & \\cdots & x_N \\times C'_M \\times (sigmoid(d5_M) \\times (1 - sigmoid(d5_M)) \\end{bmatrix} $$"

const dwilatecode = "#dwi\ndEwrtctprev = dEwrtctprev * np.transpose(self.ftstorage[t + 1])\ndEwrtit = dEwrtctprev * np.transpose(self.ctdstorage[t])\ndEwrtdi = dEwrtit * np.transpose(activation.sigmoid(self,self.itinputstorage[t]) * (1 - activation.sigmoid(self,self.itinputstorage[t])) )\ndEwrtWi += np.transpose(dEwrtdi) * np.transpose(self.input)"


const tequal3computehighlightedwf = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\underline{ \\textcolor{blue}{ \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial C_{t-2}}{\\partial a_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot(\\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus { \\frac{\\partial C_{t-3}}{\\partial a_{t-3}}  \\odot \\frac{\\partial a_{t-3}}{\\partial F_{t-3}} \\odot \\frac{\\partial F_{t-3}}{\\partial d1_{t-3}} \\odot \\frac{\\partial d1_{t-3}}{\\partial d2_{t-3}} \\odot \\frac{\\partial d2_{t-3}}{\\partial W_f}}) \\oplus { \\frac{\\partial C_{t-2}}{\\partial a_{t-2}}  \\odot \\frac{\\partial a_{t-2}}{\\partial F_{t-2}} \\odot \\frac{\\partial F_{t-2}}{\\partial d1_{t-2}} \\odot \\frac{\\partial d1_{t-2}}{\\partial d2_{t-2}} \\odot \\frac{\\partial d2_{t-2}}{\\partial W_f}} )\\oplus { \\frac{\\partial C_{t-1}}{\\partial a_{t-1}}  \\odot \\frac{\\partial a_{t-1}}{\\partial F_{t-1}} \\odot \\frac{\\partial F_{t-1}}{\\partial d1_{t-1}} \\odot \\odot \\frac{\\partial d1_{t-1}}{\\partial d2_{t-1}} \\odot \\frac{\\partial d2_{t-1}}{\\partial W_f}} ) \\oplus  { \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}} ) $$"
const rightmosttermwf = "$$ \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f} = 1 \\odot diag(C_{t-1}) \\odot (sigmoid(d1_t) \\times (1 - sigmoid(d1_t))) \\odot \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} $$"
const dCtoverdWfcont = "$$ \\begin{bmatrix} x_1 \\times diag(C_{{t-1}_1}) \\times (sigmoid(d1_1) \\times (1 - sigmoid(d1_1))  & x_2 \\times diag(C_{{t-1}_1}) \\times (sigmoid(d1_1) \\times (1 - sigmoid(d1_1)) & \\cdots & x_N \\times diag(C_{{t-1}_1}) \\times (sigmoid(d1_1) \\times (1 - sigmoid(d1_1)) \\\\ x_1 \\times diag(C_{{t-1}_2}) \\times (sigmoid(d1_2) \\times (1 - sigmoid(d1_2))  & x_2 \\times diag(C_{{t-1}_2}) \\times (sigmoid(d1_2) \\times (1 - sigmoid(d1_2))  & \\cdots & x_N \\times diag(C_{{t-1}_2}) \\times (sigmoid(d1_2) \\times (1 - sigmoid(d1_2))  \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\\\ x_1 \\times diag(C_{{t-1}_M}) \\times (sigmoid(d1_M) \\times (1 - sigmoid(d1_M)) & x_2 \\times diag(C_{{t-1}_M}) \\times (sigmoid(d1_M) \\times (1 - sigmoid(d1_M)) & \\cdots & x_N \\times diag(C_{{t-1}_M}) \\times (sigmoid(d1_M) \\times (1 - sigmoid(d1_M)) \\end{bmatrix} $$"

const wfpostcode = "#dwf\ndEwrtft = dEwrtctprev * np.transpose(self.ctstorage[t])\ndEwrtdf = dEwrtft * np.transpose(activation.sigmoid(self,self.itinputstorage[t]) * (1 - activation.sigmoid(self,self.itinputstorage[t])))\ndEwrtwf += np.transpose(dEwrtdf) * np.transpose(self.input)"
const wfprecode = "#dwf\ndEwrtft = dEwrtct * np.transpose(self.ctstorage[-2])\ndEwrtdf = dEwrtft * np.transpose(activation.sigmoid(self,self.finputstorage[-1]) * (1 - activation.sigmoid(self,self.finputstorage[-1])))\ndEwrtwf = np.transpose(dEwrtdf) * np.transpose(self.input)"

const wcprecode = "#dwc\ndEwrtcdt = dEwrtct * np.transpose(self.itstorage[-1])\ndEwrtcdtinp = dEwrtcdt * np.transpose(1 - np.tanh(self.ctdinpstorage[-1]) ** 2)\ndEwrtwc = np.transpose(dEwrtcdtinp) * np.transpose(self.input)\ndEwrtctprev = dEwrtct\n"

const wcpostcode = "#dwc\ndEwrtcdt = dEwrtctprev * np.transpose(self.itstorage[t])\ndEwrtcdtinp = dEwrtcdt * np.transpose(1 - np.tanh(self.ctdinpstorage[t]) ** 2)\ndEwrtwc += np.transpose(dEwrtcdtinp) * np.transpose(self.input)"

const ColoredLine = ({ color }) => (
    <hr
        style={{
            color: color,
            backgroundColor: color,
            height: 1
        }}
    />
);
return(
        <div>
          <h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <strong>Explanation of Backward Propagation code line by line</strong></h1>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;For this post, we will explore the computer science aspect of mathematical equations. Specifically, we will see how the equations expand and combine to become the line you see in python.&nbsp;</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;For readability, we recommend you open a new page from the <a href={urlredirect.derivationmainpage}>math derivation</a> and the <a href={urlredirect.LSTMfullimple}>code page</a>. This post will be dedicated to the backward function or you can see in the code</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString6}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;You can refer back and forth for formulas or math equations in the above open post.&nbsp;For this post, we will go straight to explain line by line how backward propagation work since there is a lot of Math involved.&nbsp;</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString1}
            </SyntaxHighlighter>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;One might wonder, how they come up with such equations. The answer is quite easy and straightforward. Let us begin with the math formula page.</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dEoverdwv}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As you have seen online, there are many loss functions to choose from, most notable is a softmax with cross-entropy. However, since our input and output are an array, we choose to use the <strong>Mean Square Error</strong> loss function for estimation since we are trying to compute the output array from the input array.&nbsp;&nbsp;</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dEoverdG_backward}
                        {dEoverdG_backward_cont3}
                        {dEoverdG_backward_cont5}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;From the original formula, the derivative for the Mean Square Error is intuitively <strong>(Lt - Yt)&nbsp;</strong>since the power rule is&nbsp;</p>
<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {powerrule}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;So it makes sense that the line of code&nbsp;</p>

<SyntaxHighlighter language="python" style={dracula}>
                {codeString1}
            </SyntaxHighlighter>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;is the representation of the derivative. This is the most important step since it lays out the foundation for future steps for different weight gradients.</p>

<p>&nbsp;</p>

<p><span style={{fontSize:32.04}}>An explanation for <strong>Wv</strong> implementation</span></p>
<ColoredLine color="#E3E3E3" />

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Let us start with the gradient for&nbsp;<strong>Wv&nbsp;</strong>matrix. As we know from the implementation, the <strong>Wv&nbsp;</strong>is computed as the following</p>

<SyntaxHighlighter language="python" style={dracula}>
                {Wvprecode}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Let us open the <strong>Wv</strong> derivation <a href={urlredirect.Wvderivationpage}>page</a>. The final result requires we multiply the partial derivative of the loss function with the partial derivative of the logits function. Let us observe the following multiplication.</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {Wvlater}
                        {dEoverdWv}
                        {dEoverdWvpre}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; If you have a sharp eye, you might notice above that the matrix is of the dimension 1 x MT and it might not make sense to update the <strong>Wv&nbsp;</strong>matrix with 1 x MT since <strong>Wv&nbsp;</strong>is a T x M. But since we are using SGD(Stochastic Gradient Descent), it is necessary to transform 1 x MT to TxM. The transformation process is easy, we just want to make a matrix with a starting point of h<span style={{fontSize:10.68}}>1</span>&nbsp;and an ending point of h<span style={{fontSize:10.68}}>m</span>, which means we need to have columns of M element and T rows. The final matrix will look like this.</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dEoverdWvfinal}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Since we are trying to make a <strong>TxM </strong>matrix, we need to find a way to make it. Here we use transpose of h because h computed from the forward step is a&nbsp;&nbsp;<strong>1 x M&nbsp;</strong>matrix, we need to transpose it to<strong>&nbsp;M x 1</strong>&nbsp;and multiply it with the <strong>dEwrtdL, </strong>which is&nbsp;&nbsp;<strong>&part;E(Pi, Lᵢ,)/&part;Lt</strong>.&nbsp;and has a dimension <strong>T X 1.&nbsp;</strong>It is easy to demonstrate why this can make a<strong> T X M</strong> matrix. You can copy and paste below code</p>


<SyntaxHighlighter language="python" style={dracula}>
                {transposedemonstration}
            </SyntaxHighlighter>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; As for the&nbsp;<strong>Wv</strong> backpropagation, there is no such thing because <strong>Wv&nbsp;</strong>does not depend on the time step computation. It does not go through forward propagation like <strong>Wi, Wc</strong>&nbsp;and etc. So we are safe to ignore the time back propagation here.</p>

<p>&nbsp;</p>

<p><span style={{fontSize:32.04}}>An explanation for <strong>Wo</strong> implementation</span></p>
<ColoredLine color="#E3E3E3" />

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Next, let us move with the gradient for <strong>Wo&nbsp;</strong>matrix. For this section, we recommend you open up the <strong>Wo</strong> <a href={urlredirect.Woderivationpage}>derivation</a> page. As we know from the implementation, the&nbsp;<strong>Wo&nbsp;</strong>is computed as the following from the python code</p>

<SyntaxHighlighter language="python" style={dracula}>
                {woprecode}
            </SyntaxHighlighter>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As we know, in order to compute <strong>Wo</strong>, we follow a different equation set. The equation set is the following.</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dEoverdWosimplifiedcancledcont}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;The first three-term has already been computed in the above, please feel free to check out&nbsp;<strong>Wv </strong>above. Here we will just expand the rest of the equations. Here we will not multiply the matrix together since it expands too many terms. We have just to know that it is a dot product, which Numpy will handle respectively.&nbsp;</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dEoverdht}
                        {dLoverdht}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;The next we will compute is the rest of the above equations.</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dhtoverWo}
                        {dhtoverWopre}

                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Since the <strong>Wo&nbsp;</strong>is again <strong>M x N </strong>matrix, we need to transform the <strong>1 X MN</strong> matrix to <strong>M x N&nbsp;</strong>in order to make them do SGD. So the final form after transforming is</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dhtoverWocont}

                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As you can see above, &quot;self. b&quot; is equivalent to our general formula, the&nbsp;<strong>&sigma;(C<span style={{fontSize:12.015}}>t</span>)</strong>.&nbsp;<strong>&nbsp;&quot;</strong>self.oinpstorage&quot; is referring the input of <strong>O<span style={{fontSize:10.68}}>t</span></strong> function in the general function and it is the same function that drives being applied in the python sigmoid derivative</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;For this function, since there are no dependencies for&nbsp;<strong>O<span style={{fontSize:10.68}}>t&nbsp;&nbsp;</span></strong>in&nbsp;<strong>C<span style={{fontSize:12.015}}>t-1</span></strong>&nbsp;o. There is time backpropagation.</p>

<p>&nbsp;</p>

<p><span style={{fontSize:32.04}}>An explanation for <strong>Wi</strong> implementation</span></p>
<ColoredLine color="#E3E3E3" />

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Next, let us move with the function for <strong>Wi</strong>. For this section, we recommend you open up the <strong>Wi</strong>&nbsp;<a href={urlredirect.Widerivationpage}>derivation</a> page. As we know from the implementation, the&nbsp;<strong>Wi&nbsp;</strong>is computed as the following from the python code</p>

<SyntaxHighlighter language="python" style={dracula}>
                {wiprecode}
            </SyntaxHighlighter>


<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;We can reuse a lot of terms above, specifically the&nbsp;<strong>&part;E/&part;h<span style={{fontSize:10.68}}>t</span></strong>, this term is already computed so we can just use it to compute other matrix gradients. There will be two parts to this matrix gradient, one is the constant, and the other is the time propagation. Let us start with the constant first.</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {ddoverd1timesd21overdWocombine1}
                        {tequal3computehighlighted}

                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;The next should be the rightmost terms, which are similar to those above and need to be transformed to <strong>M X N</strong> matrix, which then is Gradient Descended from the original weight matrix. We will not go through some expansion and will instead write down the final result here.</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {dEoverdCt}
                        {dCtoverdWi}
                        {dCtoverdWicont}

                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;It should be intuitive to make the connection here between the mathematical derivation and the python code implementation. The&nbsp;<strong>dEwrtdht</strong> is the previously computed value between <strong>W<span style={{fontSize:10.68}}>v</span></strong>&nbsp;and the derivative of the loss function. The &quot;<strong>np.transpose(self.ostorage[-1])</strong>&quot; mean we are transposing the input from the <strong>O<span style={{fontSize:10.68}}>t&nbsp;</span></strong>and the reason we are transposing is because the output from the &quot;<strong>self.ostorage</strong>&quot; is a <strong>M x 1</strong>&nbsp;matrix and we want to make it to <strong>1 x M&nbsp;</strong>so that we can make it like a diag matrix and we don&#39;t have to make an identity matrix and make it a diag. The end result will be <strong>1 x M</strong> anyway. The transpose mechanics can be applied to other equations as well. The rest will be the same &quot;<strong>self.ctstorage</strong>&quot; is the <strong>Ct&nbsp;</strong>computed from the forward steps and the&nbsp;&quot;<strong>self.ctdstorage</strong>&quot; is the&nbsp;<strong>C&#39;_t&nbsp;</strong>computed from the forward steps. The &quot;self.itinputstorage&quot; is the input parameter from&nbsp;<strong>i<span style={{fontSize:10.68}}>t</span></strong>. Lastly, we are multiplying the transpose of <strong>&quot;dEwrtdi&quot;</strong> and the transpose of <strong>&quot;self.input&quot;&nbsp;</strong>because we want to make sure that the we are making a M X N matrix. In Numpy, <strong>M x 1</strong> with <strong>1 x N</strong> is <strong>M x N</strong>.</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;The next will be the time propagation steps. Let us start with the code</p>


<SyntaxHighlighter language="python" style={dracula}>
                {dwilatecode}
            </SyntaxHighlighter>


<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As you can see, the extra equation between this python code and the above python code is the <strong>&quot;dEwrtctprev = dEwrtctprev * np.transpose(self.ftstorage[t + 1])&quot;.&nbsp;</strong>It would not take long to realize that &quot;<strong>self.ftstorage</strong>&quot; is the blue-term product in the timestep function above. The rest of the functions in the time propagation steps function are repeats of multiplication computed from the other function above except that it is older in the storage from the forward computation.</p>

<p>&nbsp;</p>

<p><span style={{fontSize:32.04}}>An explanation for <strong>Wf</strong> implementation</span></p>
<ColoredLine color="#E3E3E3" />

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<strong>Wf&nbsp;</strong>is what we want to compute next. As usual, let us write down the non-time propagation python code for&nbsp;<strong>Wf</strong>.&nbsp;</p>
<SyntaxHighlighter language="python" style={dracula}>
                {wfprecode}
            </SyntaxHighlighter>


<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As usual, let us write down the formula. We recommend you open up the <strong>Wf</strong>&nbsp;<a href={urlredirect.Wfderivationpage}>derivation</a> page to see all the derivations. But for this post, let us just write down the last formula from the page.</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {tequal3computehighlightedwf}

                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As you can see from the python code above, we are reusing the term &quot;<strong>dEwrtct</strong>&quot;, which makes sense because if you look at the formula above, the terms leading to the time backpropagation step are the same as that in&nbsp;<strong>Wi</strong>. So we should be safe to reuse computation from there. Now let us just compute the rightmost term. Since it is similar to the above situation, where we need to transform the matrix to fit the dimension of&nbsp;<strong>Wf,&nbsp;</strong>we will just write down the equation here</p>

<MathJaxContext config={config} version={3}>
                    <MathJax inline>     
                        {rightmosttermwf}
                        {dCtoverdWfcont}

                    </MathJax>
                </MathJaxContext>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; As you can see the math formula above matches exactly how python code is implemented. For example, &quot;<strong>self.finputstorage</strong>&quot; is referring to the d1 in the forget gate function. So the python is just the math derivation. The rest you can look at the formula and figure out.</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Now, let us move to the time propagation step. First let us check out the python code&nbsp;</p>

<SyntaxHighlighter language="python" style={dracula}>
                {wfpostcode}
            </SyntaxHighlighter>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Notice, we are again reusing the &quot;<strong>dEwrtctprev</strong>&quot;, which is computed above from the&nbsp;<strong>Wi</strong>. The reason we can reuse is very easy to understand, all you have to do is to look at the math formula on the derivation page and notice that the new additional terms are indifferent from the one from&nbsp;<strong>Wi.&nbsp;</strong>And again next, we have to do the same code above in the older timestep and reuse the stuff computed in the same old timestep.&nbsp;</p>

<p>&nbsp;</p>

<p><span style={{fontSize:32.04}}>An explanation for <strong>Wc</strong> implementation</span></p>
<ColoredLine color="#E3E3E3" />

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Since this post is dedicated toward computer science and computer science is an art of simplicity. A lot of computations have already been computed. All you have to do is to look at&nbsp;<strong>Wc&nbsp;</strong>derivation page and figure out the logic. The&nbsp;<strong>Wc</strong>&nbsp;derivation follows exactly how&nbsp;<strong>Wf&nbsp;</strong>is computed and it should be fairly easy to understand what is going on. So in this final section, we will just write down the final matrix and python code</p>

<SyntaxHighlighter language="python" style={dracula}>
                {wcprecode}
                {wcpostcode}
            </SyntaxHighlighter>

<p>&nbsp;</p>
  


        </div>
      )

}

export default LSTMbackward;