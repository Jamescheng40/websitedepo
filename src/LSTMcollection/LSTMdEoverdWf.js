import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';
function LSTMdEoverdWf(){
    const config = {
        "HTML-CSS":{ 
            scale: 150
        }
      };

    //dEoverdwf
    const Ftwf = '$$ {F_t = }\\sigma_s{(\\sum_{i=0}^M W_f \\otimes x_t + bias(optional))} = \\sigma_s{( W_f \\odot x_t + bias(optional))}{= \\sigma_s{(d)} \\; {,where \\; d=\\sigma_s(\\sum_{i=0}^M W_f \\otimes x_t) \\; is \\; sigmoid \\; and \\; d1=\\sum_{i=0}^M d2}, \\; d2=W_f \\otimes x_t}$$'; 
    const itwf = '$$ {i_t = }\\sigma_s{(\\sum_{i=0}^M W_i \\otimes x_t + bias(optional))} = \\sigma_s{( W_i \\odot x_t + bias(optional))} = \\sigma_s{(d5)} \\; {where \\; d5=\\sigma_s(\\sum_{i=0}^M W_i \\otimes x_t) \\; is \\; sigmoid \\; and \\; d6=\\sum_{i=0}^M d7}, \\; d7=W_i \\otimes x_t   $$';
    const Otwf = '$$ {O_t = }\\sigma_s{(\\sum_{i=0}^M W_o \\otimes x_t + bias(optional))} = \\sigma_s{( W_o \\odot x_t + bias(optional))} $$';
    const Cdtwf = "$$ {C'_t = }\\sigma_t{(\\sum_{i=0}^M W_c \\otimes x_t + bias(optional))} = \\sigma_s{( W_c \\odot x_t + bias(optional))}{= \\sigma_s{(d10)} \\; {,where \\; d10=\\sigma_s(\\sum_{i=0}^M W_c \\otimes x_t) \\; is \\; sigmoid \\; and \\; d11=\\sum_{i=0}^M d12}, \\; d12=W_c \\otimes x_t \\; where \\; \\sigma_t \\; is \\; tanh} $$";
    const Ctwf =  "$$ {C_t = }  F_t \\otimes {(C_{t-1})} \\oplus i_t \\otimes C'_t {, \\; where \\; \\otimes \\; is \\; element-wise \\; multiply \\; and \\oplus \\; is \\; element-wise \\; addition  ,where \\; a=F_t \\otimes {(C_{t-1})} \\; and \\; a1= i_t \\otimes C'_t }$$";
    const htwf  = "$$ {h_t = }  O_t \\otimes \\sigma_t{(C_t)} = {O_t \\otimes b_t} {, \\; where \\; b_t \\; is \\; \\sigma_t{(C_t)}} $$";
    const Liwf = "$$ {L_t =  W_v \\odot h_t =}{\\sum_{i=0}^T W_v \\otimes h_t + bias(optional)}, \\; where \\; L1 =\\sum_{i=0}^M L2,and \\; L2=W_v \\otimes h_t $$"
    const Piwf = "$$ {P_i = softmax(L_i)}{=\\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}} } $$"
    const Eiwf = "$$ {E_i=}{- \\sum_{i=0}^T Y_i  \\otimes log(P_i) + bias(optional)}{= - \\sum_{i=0}^T g,where \\; g = Y_i  \\otimes log(P_i)} $$"

    const dEoverdWf = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_f} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f} \\oplus \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_f} ) \\oplus \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_f}))}    $$";
  
    const dEoverdWfcrossedout = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_f}} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f} \\oplus \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_f} ) \\oplus \\cancelto{0}{\\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_f}))}}    $$";

    const dEoverdWfexpanded = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_f}} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f} \\oplus \\underline{\\textcolor{blue}{\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d} \\odot \\frac{\\partial d}{\\partial d1} \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}}}) \\oplus \\cancelto{0}{\\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_f}))}}    $$";

    const dEoverdWfsimplified = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot (  \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f} \\oplus { \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d} \\odot \\frac{\\partial d}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}}) }    $$";

    const computefirst = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot (  \\cancelto{0}{\\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f}} \\oplus {\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}}) }       $$";

    const computefirstcont = "$$ {= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot  {\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}}} $$"

    const computelater = "$$ \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f}  $$"
    const dEoverdG_j_final = "$$ \\frac{\\partial E}{\\partial g_j} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times T   $$"

    const dgoverdP_i_final = "$$ \\frac{\\partial g_j}{\\partial P_i} = diag(Y_i/P_i) \\ni T \\times T  $$"

    const dPoverdLi_final = "$$\\frac{\\partial P_i}{\\partial L_i} = \\begin{bmatrix} P_1 \\times (1 - P_1) & - P_1 \\times P_2 & \\cdots & -P_1 \\times P_T \\\\ -P_2 \\times P_1 & P_2 \\times (1 - P_2) & \\cdots & P_2 \\times P_T  \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ -P_T \\times P_1 & -P_T \\times P_2 & \\cdots & P_T \\times (1 - P_T) \\end{bmatrix} = P_i \\times (\\delta_{ij} - P_j ) $$"
    const kronica = "$$ \\delta_{ij} = \\begin{cases}1, &         \\text{if } i=j,\\\\0, &  \\text{if } i\\neq j.\\end{cases} $$"
    const dL2overdhfinalproduct = "$$ \\frac{\\partial L2_1}{\\partial h_t} = \\begin{bmatrix} W_{{v}_{11}} & 0 & \\cdots & 0 \\\\ 0 & W_{{v}_{12}}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{{v}_{1M}} \\end{bmatrix} = diag(W_{{v}_{1i}}) \\ni T \\times M $$"
    const dL22overdh2finalproduct = "$$\\frac{\\partial L2_2}{\\partial h_t} =\\begin{bmatrix} W_{{v}_{21}} & 0 & \\cdots & 0 \\\\ 0 & W_{{v}_{22}}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{{v}_{2M}} \\end{bmatrix} = diag(W_{{v}_{2i}}) \\ni T \\times M  $$"
    const dL21overdh2timesdL2overdhfinalproduct = "$$ \\frac{\\partial L1_1}{\\partial L2_1} \\odot \\frac{\\partial L2_1}{\\partial h_t} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} W_{{v}_{11}} & 0 & \\cdots & 0 \\\\ 0 & W_{{v}_{12}}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{{v}_{1M}} \\end{bmatrix} = \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}}  \\end{bmatrix}  $$"
    const dL22overdh2timesdL2overdhfinalproduct = "$$\\frac{\\partial L1_2}{\\partial L2_2} \\odot \\frac{\\partial L2_2}{\\partial h_t}  = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} W_{{v}_{21}} & 0 & \\cdots & 0 \\\\ 0 & W_{{v}_{22}}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{{v}_{2M}} \\end{bmatrix} =  \\begin{bmatrix} W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}}  \\end{bmatrix} $$"
    const dL2ioverdh2ifinalproduct = "$$ \\frac{\\partial Li}{\\partial h_t} = \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix} \\ni T \\times M $$"
    
    
    const datoverdbt = "$$ \\frac{\\partial a}{\\partial f_t} =  \\frac{\\partial \\begin{bmatrix} f_{{t}_1} \\times C_{{t-1}_1} \\\\ f_{{t}_2} \\times C_{{t-1}_2} \\\\ \\vdots \\\\ f_{{t}_M} \\times C_{{t-1}_M} \\end{bmatrix}}{\\partial P_i} = \\begin{bmatrix} \\frac{\\partial (f_{{t}_1} \\times C_{{t-1}_1})}{\\partial f_{{t}_1}} & \\cancelto{0}{\\frac{\\partial (f_{{t}_1} \\times C_{{t-1}_1})}{\\partial f_{{t}_2}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (f_{{t}_1} \\times C_{{t-1}_1})}{\\partial f_{{t}_M}}} \\\\ \\cancelto{0}{\\frac{\\partial (f_{{t}_2} \\times C_{{t-1}_2})}{\\partial f_{{t}_1}}} & \\frac{\\partial (f_{{t}_2} \\times C_{{t-1}_2})}{\\partial f_{{t}_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (f_{{t}_2} \\times C_{{t-1}_2})}{\\partial f_{{t}_M}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (f_{{t}_M} \\times C_{{t-1}_M})}{\\partial f_{{t}_M}}} & \\cancelto{0}{\\frac{\\partial (f_{{t}_M} \\times C_{{t-1}_M})}{\\partial f_{{t}_2}}} & \\cdots & \\frac{\\partial (f_{{t}_M} \\times C_{{t-1}_M})}{\\partial f_{{t}_M}}  \\end{bmatrix} = diag(C'_{t-1}) \\ni M \\times M $$"

    const ditoverdd1 = "$$\\frac{\\partial \\begin{bmatrix} sigmoid(d_1) \\\\ sigmoid(d_2) \\\\ \\vdots \\\\ sigmoid(d_M) \\end{bmatrix}}{\\partial d_t} = \\begin{bmatrix} \\frac{\\partial (sigmoid(d_1))}{\\partial d_1} & \\cancelto{0}{\\frac{\\partial (sigmoid(d_1))}{\\partial d_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (sigmoid(d_1))}{\\partial d_M}} \\\\ \\cancelto{0}{\\frac{\\partial (sigmoid(d_2))}{\\partial d_1}} & \\frac{\\partial (sigmoid(d_2))}{\\partial d_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (sigmoid(d_2))}{\\partial d_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (sigmoid(d_M))}{\\partial d_1}} & \\cancelto{0}{\\frac{\\partial (sigmoid(d_M))}{\\partial d_2}} & \\cdots & \\frac{\\partial (sigmoid(d_M))}{\\partial d_M}  \\end{bmatrix} = sigmoid(d_t) \\times (1 - sigmoid(d_t)) \\ni M \\times M $$"
    const dsigmoid1 = "$$ \\frac{\\partial (sigmoid(d_1))}{\\partial d_1} = sigmoid(d_1) \\times (1 - sigmoid(d_1)) $$"
    const ddoverdd1 = "$$ \\frac{\\partial d}{\\partial d1} = 1 $$"
    const dd1overdd2 = "$$ $$"


    const ddoverd11 = "$$ \\frac{\\partial d1_1}{\\partial d2_1} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N; \\; \\frac{\\partial d1_2}{\\partial d2_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N \\cdots \\frac{\\partial d1_N}{\\partial d2_N} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N $$"

    const d21overdWif = "$$\\frac{\\partial \\begin{bmatrix} W_{{f}_{11}} \\times x_1 \\\\ W_{{f}_{12}} \\times x_2 \\\\ \\vdots \\\\ W_{{f}_{1N}} \\times x_N \\end{bmatrix}}{\\partial W_f} = \\begin{bmatrix} \\frac{\\partial ( W_{{f}_{11}} \\times x_1 )}{\\partial W_{{f}_{11}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{11}} \\times x_1)}{\\partial W_{{f}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{11}} \\times x_1)}{\\partial W_{{f}_{1N}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{11}} \\times x_1)}{\\partial W_{{f}_{MN}}}} & \\cdots & 0 \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{11}}}} & \\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{1N}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{MN}}}} & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{f}_{1N}} \\times x_N)}{\\partial W_{{f}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{1N}} \\times x_N)}{\\partial W_{{f}_{12}}}} & \\cdots & \\frac{\\partial (W_{{f}_{1N}} \\times x_N)}{\\partial W_{{f}_{1N}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{1N}} \\times x_N)}{\\partial W_{{f}_{MN}}}} & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix}  =  \\ni M \\times M $$ "
    const d22overdWif = "$$\\frac{\\partial \\begin{bmatrix} W_{{f}_{11}} \\times x_1 \\\\ W_{{f}_{12}} \\times x_2 \\\\ \\vdots \\\\ W_{{f}_{1N}} \\times x_N \\end{bmatrix}}{\\partial W_f} = \\begin{bmatrix} \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{1N}}}} & \\frac{\\partial ( W_{{f}_{21}} \\times x_1 )}{\\partial W_{{f}_{21}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{22}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{MN}}}}  \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{11}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{1N}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{21}}}} & \\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{22}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{MN}}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{1N}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{21}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{22}}}} & 0 & \\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{2N}}} & \\cdots &  \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{MN}}}}  \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix}  =  \\ni N \\times MN $$ "
    
    const ddoverd1timesd21overdWf = "$$ \\frac{\\partial d1_1}{\\partial d2_1} \\odot \\frac{\\partial d2_1}{\\partial Wi} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
    const ddoverd12timesd21overdWf = "$$ \\frac{\\partial d1_2}{\\partial d2_2} \\odot \\frac{\\partial d2_2}{\\partial Wi} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
    const ddoverd1timesd21overdWfcombine1 = "$$ \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} \\ni M \\times MN  $$"
    
    const tequal3computenohighlightedwf = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( { \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}} \\odot ( {\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}} \\odot ( {{\\frac{\\partial C_{t-2}}{\\partial a_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}} \\odot(\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}} \\oplus { \\frac{\\partial C_{t-3}}{\\partial a_{t-3}}  \\odot \\frac{\\partial a_{t-3}}{\\partial F_{t-3}} \\odot \\frac{\\partial F_{t-3}}{\\partial d_{t-3}} \\odot \\frac{\\partial d_{t-3}}{\\partial d1_{t-3}} \\odot \\frac{\\partial d1_{t-3}}{\\partial d2_{t-3}} \\odot \\frac{\\partial d2_{t-3}}{\\partial W_f}}) \\oplus { \\frac{\\partial C_{t-2}}{\\partial a_{t-2}}  \\odot \\frac{\\partial a_{t-2}}{\\partial F_{t-2}} \\odot \\frac{\\partial F_{t-2}}{\\partial d_{t-2}} \\odot \\frac{\\partial d_{t-2}}{\\partial d1_{t-2}} \\odot \\frac{\\partial d1_{t-2}}{\\partial d2_{t-2}} \\odot \\frac{\\partial d2_{t-2}}{\\partial W_f}} )\\oplus { \\frac{\\partial C_{t-1}}{\\partial a_{t-1}}  \\odot \\frac{\\partial a_{t-1}}{\\partial F_{t-1}} \\odot \\frac{\\partial F_{t-1}}{\\partial d_{t-1}} \\odot \\frac{\\partial d_{t-1}}{\\partial d1_{t-1}} \\odot \\odot \\frac{\\partial d1_{t-1}}{\\partial d2_{t-1}} \\odot \\frac{\\partial d2_{t-1}}{\\partial W_f}} ) \\oplus  { \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d} \\odot \\frac{\\partial d}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}} ) $$"
    const tequal3computehighlightedwf = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\underline{ \\textcolor{blue}{ \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial C_{t-2}}{\\partial a_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot(\\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus { \\frac{\\partial C_{t-3}}{\\partial a_{t-3}}  \\odot \\frac{\\partial a_{t-3}}{\\partial F_{t-3}} \\odot \\frac{\\partial F_{t-3}}{\\partial d_{t-3}} \\odot \\frac{\\partial d_{t-3}}{\\partial d1_{t-3}} \\odot \\frac{\\partial d1_{t-3}}{\\partial d2_{t-3}} \\odot \\frac{\\partial d2_{t-3}}{\\partial W_f}}) \\oplus { \\frac{\\partial C_{t-2}}{\\partial a_{t-2}}  \\odot \\frac{\\partial a_{t-2}}{\\partial F_{t-2}} \\odot \\frac{\\partial F_{t-2}}{\\partial d_{t-2}} \\odot \\frac{\\partial d_{t-2}}{\\partial d1_{t-2}} \\odot \\frac{\\partial d1_{t-2}}{\\partial d2_{t-2}} \\odot \\frac{\\partial d2_{t-2}}{\\partial W_f}} )\\oplus { \\frac{\\partial C_{t-1}}{\\partial a_{t-1}}  \\odot \\frac{\\partial a_{t-1}}{\\partial F_{t-1}} \\odot \\frac{\\partial F_{t-1}}{\\partial d_{t-1}} \\odot \\frac{\\partial d_{t-1}}{\\partial d1_{t-1}} \\odot \\odot \\frac{\\partial d1_{t-1}}{\\partial d2_{t-1}} \\odot \\frac{\\partial d2_{t-1}}{\\partial W_f}} ) \\oplus  { \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d} \\odot \\frac{\\partial d}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}} ) $$"


        const dctovera = "$$ \\frac{\\partial C_t}{\\partial a_t} = 1 $$";

        const daoverdctm1 = "$$ \\frac{\\partial a_{t}}{C_{t-1}} = \\frac{\\partial (F_t \\otimes C_{t-1})}{\\partial C_{t-1}} = diag(F_t) \\ni 1 \\times M  $$";

        const da1overdctm2 = "$$ \\frac{\\partial a_{t-1}}{C_{t-2}} = \\frac{\\partial (F_{t-1} \\otimes C_{t-2})}{\\partial C_{t-2}} = diag(F_{t-1}) \\ni 1 \\times M  $$";
        const dctm1overdatm1 =  "$$ \\frac{\\partial C_{t-1}}{\\partial a_{t-1}} = 1 $$"
      return (
        
        <div>
            <h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <strong>Full derivation of&nbsp;&part;E/&part;Wf</strong></h1>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Same as the other posts, we will write down the basic equation and dimensionality first as a reference</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Dimensionality: M is defined as a hidden unit number, N is the input element number, T is the output element number</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As stated before, the summation is just the row of the weight matrix multiplied by the column of the input vector. And it represents one hidden unit out of M.</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Just remember that each page has different symbols such as a1 that are dedicated toward the local page only and never apply them to a different page.</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
        {Ftwf}
        {itwf}
        {Otwf}
        {Cdtwf}
        {Ctwf}
        {htwf}
        {Liwf}
        {Piwf}
        {Eiwf}
    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Next, as usual, let us write down the full derivation formula</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
       {dEoverdWf}
    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Next, let us expand on some of the terms that are not zero. In particular, we have highlighted the term that needs expansion</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
       {dEoverdWfcrossedout}
       {dEoverdWfexpanded}
       {dEoverdWfsimplified}
    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Here like before, let us first compute the constant term which has the timestep set to 1 or t = 1, the term will reduce to this</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
      {computefirst}
      {computefirstcont}
    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The first part in blue has already been computed please check out this <a href="http://post">post</a> or this <a href="http://post1">post</a>&nbsp;for the full derivation of those equations. For this post, we will keep it simple and use only the final product from the derivation. The derivation is based on the softmax and logit. Those are needed in order to compute the scaled output from hidden units. In the original LSTM paper, it only has one output so their T is assumed to be one. But here we assume the output is T.</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
      {dEoverdG_j_final}
      {dgoverdP_i_final}
      {dPoverdLi_final}
      {kronica}
      {dL2overdhfinalproduct}
      {dL22overdh2finalproduct}
      {dL21overdh2timesdL2overdhfinalproduct}
      {dL22overdh2timesdL2overdhfinalproduct}
      {dL2ioverdh2ifinalproduct}

    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As mentioned above, since we assumed the output contains T elements, the weight matrix should be stacked upto T elements</p>

<p>&nbsp;</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Let us continue to derive other equations</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
      {datoverdbt}
      {ditoverdd1}
      {dsigmoid1}


    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; It should be noted that variable d above is different from the other page&#39;s variable d. In this instance, the d is the input equation for the sigmoid for the Ft</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
      {ddoverdd1}
      {dd1overdd2}
      {ddoverd11}
      {d21overdWif}
      {d22overdWif}
      {ddoverd1timesd21overdWf}
      {ddoverd12timesd21overdWf}
      {ddoverd1timesd21overdWfcombine1}
      </MathJax>
</MathJaxContext>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Since we assumed there are M hidden units, we should stack the matrix upto M rows</p>

<p>&nbsp;</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Now let us try to see if we are at a different timestep and how the equations are going to expand out. Some of the equations would go deeper than initially would be. Let us see how this expanded out.</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
       {tequal3computenohighlightedwf}
       {tequal3computehighlightedwf}
    </MathJax>
</MathJaxContext>


<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice that there are extra 2 partial differential equations at the beginning of the blue color highlight. This is the only exception for Wf, and the rest for t=4, t=5, t=6 will have the same rate of adding two extra equations per line.</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
     {dctovera}
     {daoverdctm1}
     {dctm1overdatm1}
     {da1overdctm2}
    </MathJax>
</MathJaxContext>

        </div>
    )
}
  
export default LSTMdEoverdWf;