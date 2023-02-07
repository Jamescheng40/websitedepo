import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';
function LSTMdEoverdWf(){
    const config = {
        "HTML-CSS":{ 
            scale: 150
        }
      };

    //dEoverdwf
    const Ftgeneral = '$$ {F_t = }\\sigma_s{(\\sum_{i=0}^M W_f \\otimes x_t + bias(optional))} = \\sigma_s{( W_f \\odot x_t + bias(optional))}{= \\sigma_s{(d1)} \\; {,where \\; d1=\\sum_{i=0}^M W_f \\otimes x_t \\; is \\; sigmoid \\; and \\; d1=\\sum_{i=0}^M d2}, \\; d2=W_f \\otimes x_t}$$'; 
    const itgeneral = '$$ {i_t = }\\sigma_s{(\\sum_{i=0}^M W_i \\otimes x_t + bias(optional))} = \\sigma_s{( W_i \\odot x_t + bias(optional))} = \\sigma_s{(d6)} \\; {where \\; d6=\\sum_{i=0}^M W_i \\otimes x_t \\; is \\; sigmoid \\; and \\; d6=\\sum_{i=0}^M d7}, \\; d7=W_i \\otimes x_t   $$';
    const Otgeneral = '$$ {O_t = }\\sigma_s{(\\sum_{i=0}^M W_o \\otimes x_t + bias(optional))} = \\sigma_s{( W_o \\odot x_t + bias(optional)) = \\sigma_s{(d13)}} {where \\; d13=\\sum_{i=0}^M W_O \\otimes x_t \\; is \\; sigmoid \\; and \\; d13=\\sum_{i=0}^M d14}, \\; d14=W_o \\otimes x_t  $$';
    const Cdtgeneral = "$$ {C'_t = }\\sigma_t{(\\sum_{i=0}^M W_c \\otimes x_t + bias(optional))} = \\sigma_t{( W_c \\odot x_t + bias(optional))}{= \\sigma_t{(d11)} \\; {,where \\; d11=\\sum_{i=0}^M W_c \\otimes x_t \\; is \\; sigmoid \\; and \\; d11=\\sum_{i=0}^M d12}, \\; d12=W_c \\otimes x_t \\; where \\; \\sigma_t \\; is \\; tanh} $$";
    const Ctgeneral =  "$$ {C_t = }  F_t \\otimes {(C_{t-1})} \\oplus i_t \\otimes C'_t {, \\; where \\; \\otimes \\; is \\; element-wise \\; multiply \\; and \\oplus \\; is \\; element-wise \\; addition  ,where \\; a=F_t \\otimes {(C_{t-1})} \\; and \\; a1= i_t \\otimes C'_t }$$";
    const htgeneral  = "$$ {h_t = }  O_t \\otimes \\sigma_t{(C_t)} = {O_t \\otimes b_t} {, \\; where \\; b_t \\; is \\; \\sigma_t{(C_t)}} $$";
    const Ligeneral = "$$ {L_t =  W_v \\odot h_t =}{\\sum_{i=0}^T W_v \\otimes h_t + bias(optional)}, \\; where \\; L1 =\\sum_{i=0}^M L2,and \\; L2=W_v \\otimes h_t $$"
    //const Pigeneral = "$$ {P_i = softmax(L_i)}{=\\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}} } $$"
    const Eigeneral = "$$ {E_i=}{ \\frac{1}{n} \\otimes \\sum_{i=0}^T (L_t - Y)^2  }{= \\frac{1}{n} \\otimes \\sum_{i=0}^T g,where \\; g = (L_t - Y)^2}.\\; The \\; Loss \\; function\\; here\\; is\\; the\\; Mean \\; Squred \\; Error $$"
    
    
    const dEoverdWf = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_f} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f} \\oplus \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_f} ) \\oplus \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_f}))}    $$";
  
    const dEoverdWfcrossedout = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_f}} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f} \\oplus \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_f} ) \\oplus \\cancelto{0}{\\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_f}))}}    $$";
    
    const dEoverdWfexpanded = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_f}} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f} \\oplus \\underline{\\textcolor{blue}{\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}}}) \\oplus \\cancelto{0}{\\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_f}))}}    $$";
    
    const dEoverdWfsimplified = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot (  \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f} \\oplus { \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}}) }    $$";
    
    const computefirst = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot (  \\cancelto{0}{\\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f}} \\oplus {\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}}) }       $$";
    
    const computefirstcont = "$$ {= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot  {\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}}} $$"
    
    const computelater = "$$ \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_f}  $$"
    const dEoverdG_j_final = "$$ \\frac{\\partial E}{\\partial g_j} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times T   $$"      
    const dgoverdL_i_final = "$$ \\frac{\\partial g_j}{\\partial L_t} = \\frac{\\partial \\begin{bmatrix} (L_1 - Y_1)^2 \\\\ (L_2 - Y_2)^2 \\\\ \\vdots \\\\ (L_T - Y_T)^2 \\end{bmatrix}}{\\partial L_t} = \\begin{bmatrix} \\frac{\\partial (L_1 - Y_1)^2}{\\partial L_1} & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_1 - Y_1)^2}{\\partial L_T}} \\\\ \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_1}} & \\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_T}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}} & \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_2}} & \\cdots & \\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}  \\end{bmatrix} = diag(2 \\times (L_1 - Y_1)) \\ni T \\times T  $$"
    const dgoverdL_i_finalcont1 = "$$ \\frac{\\partial (L_1 - Y_1)^2}{\\partial L_1} = 2 \\times (L_1 - Y_1) \\times (\\cancelto{1}{\\frac{\\partial L_1}{\\partial L_1}} - \\cancelto{0}{\\frac{\\partial Y_1}{\\partial L_1}})   $$"
    
    const dPoverdLi_final = "$$\\frac{\\partial P_i}{\\partial L_i} = \\begin{bmatrix} P_1 \\times (1 - P_1) & - P_1 \\times P_2 & \\cdots & -P_1 \\times P_T \\\\ -P_2 \\times P_1 & P_2 \\times (1 - P_2) & \\cdots & P_2 \\times P_T  \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ -P_T \\times P_1 & -P_T \\times P_2 & \\cdots & P_T \\times (1 - P_T) \\end{bmatrix} = P_i \\times (\\delta_{ij} - P_j ) $$"
    const kronica = "$$ \\delta_{ij} = \\begin{cases}1, &         \\text{if } i=j,\\\\0, &  \\text{if } i\\neq j.\\end{cases} $$"
    const dLoverdL1 = "$$ \\frac{\\partial L_i}{\\partial L1_{1}} = 1 $$"
    //const equation = '    $$\\left[ \\begin{array}{cc|c} 1&2&3\\\\ 4&5&6  \\end{array} \\right] $$ ';
    //const summation = '$$\\sigma_s{(\\sum_{i=0}^M W_f \\otimes x_t + bias(optional))} $$' 
    const dL1overdL2 = "$$ \\frac{\\partial L1_1}{\\partial L2_1} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M; \\; \\frac{\\partial L1_2}{\\partial L2_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M \\cdots \\frac{\\partial L1_T}{\\partial L2_T} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M $$"
    
    const dL2overdhfinalproduct = "$$ \\frac{\\partial L2_1}{\\partial h_t} = \\begin{bmatrix} W_{{v}_{11}} & 0 & \\cdots & 0 \\\\ 0 & W_{{v}_{12}}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{{v}_{1M}} \\end{bmatrix} = diag(W_{{v}_{1i}}) \\ni T \\times M $$"
    const dL22overdh2finalproduct = "$$\\frac{\\partial L2_2}{\\partial h_t} =\\begin{bmatrix} W_{{v}_{21}} & 0 & \\cdots & 0 \\\\ 0 & W_{{v}_{22}}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{{v}_{2M}} \\end{bmatrix} = diag(W_{{v}_{2i}}) \\ni T \\times M  $$"
    const dL21overdh2timesdL2overdhfinalproduct = "$$ \\frac{\\partial L1_1}{\\partial L2_1} \\odot \\frac{\\partial L2_1}{\\partial h_t} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} W_{{v}_{11}} & 0 & \\cdots & 0 \\\\ 0 & W_{{v}_{12}}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{{v}_{1M}} \\end{bmatrix} = \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}}  \\end{bmatrix}  $$"
    const dL22overdh2timesdL2overdhfinalproduct = "$$\\frac{\\partial L1_2}{\\partial L2_2} \\odot \\frac{\\partial L2_2}{\\partial h_t}  = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} W_{{v}_{21}} & 0 & \\cdots & 0 \\\\ 0 & W_{{v}_{22}}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{{v}_{2M}} \\end{bmatrix} =  \\begin{bmatrix} W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}}  \\end{bmatrix} $$"
    const dL2ioverdh2ifinalproduct = "$$ \\frac{\\partial Li}{\\partial h_t} = \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} = \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix} \\ni T \\times M $$"
    
    
    const datoverdbt = "$$ \\frac{\\partial a}{\\partial F_t} =  \\frac{\\partial \\begin{bmatrix} F_{{t}_1} \\times C_{{t-1}_1} \\\\ F_{{t}_2} \\times C_{{t-1}_2} \\\\ \\vdots \\\\ F_{{t}_M} \\times C_{{t-1}_M} \\end{bmatrix}}{\\partial F_t} = \\begin{bmatrix} \\frac{\\partial (F_{{t}_1} \\times C_{{t-1}_1})}{\\partial F_{{t}_1}} & \\cancelto{0}{\\frac{\\partial (F_{{t}_1} \\times C_{{t-1}_1})}{\\partial F_{{t}_2}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (F_{{t}_1} \\times C_{{t-1}_1})}{\\partial F_{{t}_M}}} \\\\ \\cancelto{0}{\\frac{\\partial (F_{{t}_2} \\times C_{{t-1}_2})}{\\partial F_{{t}_1}}} & \\frac{\\partial (F_{{t}_2} \\times C_{{t-1}_2})}{\\partial F_{{t}_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (F_{{t}_2} \\times C_{{t-1}_2})}{\\partial F_{{t}_M}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (F_{{t}_M} \\times C_{{t-1}_M})}{\\partial F_{{t}_M}}} & \\cancelto{0}{\\frac{\\partial (F_{{t}_M} \\times C_{{t-1}_M})}{\\partial F_{{t}_2}}} & \\cdots & \\frac{\\partial (F_{{t}_M} \\times C_{{t-1}_M})}{\\partial F_{{t}_M}}  \\end{bmatrix} = diag(C_{t-1}) \\ni M \\times M $$"
        
    const dFtoverdd1 = "$$ \\frac{\\partial F_t}{\\partial d1_t} = \\frac{\\partial \\begin{bmatrix} sigmoid(d1_t) \\\\ sigmoid(d1_{{t}_2}) \\\\ \\vdots \\\\ sigmoid(d1_M) \\end{bmatrix}}{\\partial d1_t} = \\begin{bmatrix} \\frac{\\partial (sigmoid(d1_t))}{\\partial d1_t} & \\cancelto{0}{\\frac{\\partial (sigmoid(d1_t))}{\\partial d1_{{t}_2}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (sigmoid(d1_t))}{\\partial d1_M}} \\\\ \\cancelto{0}{\\frac{\\partial (sigmoid(d1_{{t}_2}))}{\\partial d1_t}} & \\frac{\\partial (sigmoid(d1_{{t}_2}))}{\\partial d1_{{t}_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (sigmoid(d1_{{t}_2}))}{\\partial d1_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (sigmoid(d1_M))}{\\partial d1_t}} & \\cancelto{0}{\\frac{\\partial (sigmoid(d1_M))}{\\partial d1_{{t}_2}}} & \\cdots & \\frac{\\partial (sigmoid(d1_M))}{\\partial d1_M}  \\end{bmatrix} = sigmoid(d1_t) \\times (1 - sigmoid(d1_t)) \\ni M \\times M $$"
    
    const dsigmoid1 = "$$ \\frac{\\partial (sigmoid(d_1))}{\\partial d_1} = sigmoid(d_1) \\times (1 - sigmoid(d_1)) $$"
    const ddoverdd1 = "$$ \\frac{\\partial d}{\\partial d1} = 1 $$"
    const dd1overdd2 = "$$ $$"
    
    
    const ddoverd11 = "$$ \\frac{\\partial d1_1}{\\partial d2_1} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N; \\; \\frac{\\partial d1_2}{\\partial d2_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N \\cdots \\frac{\\partial d1_N}{\\partial d2_N} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N $$"
    
    const d21overdWif = "$$ \\frac{\\partial d2_1}{\\partial W_f} = \\frac{\\partial \\begin{bmatrix} W_{{f}_{11}} \\times x_1 \\\\ W_{{f}_{12}} \\times x_2 \\\\ \\vdots \\\\ W_{{f}_{1N}} \\times x_N \\end{bmatrix}}{\\partial W_f} = \\begin{bmatrix} \\frac{\\partial ( W_{{f}_{11}} \\times x_1 )}{\\partial W_{{f}_{11}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{11}} \\times x_1)}{\\partial W_{{f}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{11}} \\times x_1)}{\\partial W_{{f}_{1N}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{11}} \\times x_1)}{\\partial W_{{f}_{MN}}}} & \\cdots & 0 \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{11}}}} & \\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{1N}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{MN}}}} & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{f}_{1N}} \\times x_N)}{\\partial W_{{f}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{1N}} \\times x_N)}{\\partial W_{{f}_{12}}}} & \\cdots & \\frac{\\partial (W_{{f}_{1N}} \\times x_N)}{\\partial W_{{f}_{1N}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{1N}} \\times x_N)}{\\partial W_{{f}_{MN}}}} & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix}  =  \\ni M \\times M $$ "
    const d22overdWif = "$$ \\frac{\\partial d2_2}{\\partial W_f} = \\frac{\\partial \\begin{bmatrix} W_{{f}_{21}} \\times x_1 \\\\ W_{{f}_{22}} \\times x_2 \\\\ \\vdots \\\\ W_{{f}_{2N}} \\times x_N \\end{bmatrix}}{\\partial W_f} = \\begin{bmatrix} \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{1N}}}} & \\frac{\\partial ( W_{{f}_{21}} \\times x_1 )}{\\partial W_{{f}_{21}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{22}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{21}} \\times x_1)}{\\partial W_{{f}_{MN}}}}  \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{11}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{1N}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{21}}}} & \\frac{\\partial ((W_{{f}_{22}} \\times x_2))}{\\partial W_{{f}_{22}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{f}_{12}} \\times x_2))}{\\partial W_{{f}_{MN}}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{1N}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{21}}}} & \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{22}}}} & 0 & \\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{2N}}} & \\cdots &  \\cancelto{0}{\\frac{\\partial (W_{{f}_{2N}} \\times x_N)}{\\partial W_{{f}_{MN}}}}  \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix}  =  \\ni N \\times MN $$ "
    
    const ddoverd1timesd21overdWf = "$$ \\frac{\\partial d1_1}{\\partial d2_1} \\odot \\frac{\\partial d2_1}{\\partial W_f} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
    const ddoverd12timesd21overdWf = "$$ \\frac{\\partial d1_2}{\\partial d2_2} \\odot \\frac{\\partial d2_2}{\\partial W_f} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
    const ddoverd1timesd21overdWfcombine1 = "$$ \\frac{\\partial d1}{\\partial W_f} = \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f} =  \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} \\ni M \\times MN  $$"
    
    const tequal3computenohighlightedwf = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot { \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}} \\odot ( {\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}} \\odot ( {{\\frac{\\partial C_{t-2}}{\\partial a_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}} \\odot(\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}} \\oplus { \\frac{\\partial C_{t-3}}{\\partial a_{t-3}}  \\odot \\frac{\\partial a_{t-3}}{\\partial F_{t-3}} \\odot \\frac{\\partial F_{t-3}}{\\partial d1_{t-3}} \\odot \\frac{\\partial d1_{t-3}}{\\partial d2_{t-3}} \\odot \\frac{\\partial d2_{t-3}}{\\partial W_f}}) \\oplus { \\frac{\\partial C_{t-2}}{\\partial a_{t-2}}  \\odot \\frac{\\partial a_{t-2}}{\\partial F_{t-2}} \\odot \\frac{\\partial F_{t-2}}{\\partial d1_{t-2}} \\odot \\frac{\\partial d1_{t-2}}{\\partial d2_{t-2}} \\odot \\frac{\\partial d2_{t-2}}{\\partial W_f}} )\\oplus { \\frac{\\partial C_{t-1}}{\\partial a_{t-1}}  \\odot \\frac{\\partial a_{t-1}}{\\partial F_{t-1}} \\odot \\frac{\\partial F_{t-1}}{\\partial d1_{t-1}} \\odot \\odot \\frac{\\partial d1_{t-1}}{\\partial d2_{t-1}} \\odot \\frac{\\partial d2_{t-1}}{\\partial W_f}} ) \\oplus  { \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}} ) $$"
    const tequal3computehighlightedwf = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\underline{ \\textcolor{blue}{ \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial C_{t-2}}{\\partial a_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot(\\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus { \\frac{\\partial C_{t-3}}{\\partial a_{t-3}}  \\odot \\frac{\\partial a_{t-3}}{\\partial F_{t-3}} \\odot \\frac{\\partial F_{t-3}}{\\partial d1_{t-3}} \\odot \\frac{\\partial d1_{t-3}}{\\partial d2_{t-3}} \\odot \\frac{\\partial d2_{t-3}}{\\partial W_f}}) \\oplus { \\frac{\\partial C_{t-2}}{\\partial a_{t-2}}  \\odot \\frac{\\partial a_{t-2}}{\\partial F_{t-2}} \\odot \\frac{\\partial F_{t-2}}{\\partial d1_{t-2}} \\odot \\frac{\\partial d1_{t-2}}{\\partial d2_{t-2}} \\odot \\frac{\\partial d2_{t-2}}{\\partial W_f}} )\\oplus { \\frac{\\partial C_{t-1}}{\\partial a_{t-1}}  \\odot \\frac{\\partial a_{t-1}}{\\partial F_{t-1}} \\odot \\frac{\\partial F_{t-1}}{\\partial d1_{t-1}} \\odot \\odot \\frac{\\partial d1_{t-1}}{\\partial d2_{t-1}} \\odot \\frac{\\partial d2_{t-1}}{\\partial W_f}} ) \\oplus  { \\frac{\\partial C_t}{\\partial a_t}  \\odot \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial d1} \\odot \\odot \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}} ) $$"
    
    
    const dctovera = "$$ \\frac{\\partial C_t}{\\partial a_t} = 1 $$";
    
    const daoverdctm1 = "$$ \\frac{\\partial a_{t}}{\\partial C_{t-1}} = \\frac{\\partial (F_t \\otimes C_{t-1})}{\\partial C_{t-1}} = diag(F_t) \\ni 1 \\times M  $$";
    
    const da1overdctm2 = "$$ \\frac{\\partial a_{t-1}}{\\partial C_{t-2}} = \\frac{\\partial (F_{t-1} \\otimes C_{t-2})}{\\partial C_{t-2}} = diag(F_{t-1}) \\ni 1 \\times M  $$";
    const dctm1overdatm1 =  "$$ \\frac{\\partial C_{t-1}}{\\partial a_{t-1}} = 1 $$"
    const dLioverdH = "$$ \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t}  $$"
    const d1overdWf  = "$$ \\frac{\\partial d1}{\\partial W_f} = \\frac{\\partial d1}{\\partial d2} \\odot \\frac{\\partial d2}{\\partial W_f}  $$"
    
    return (
        
        <div>
            <h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <strong>Full derivation of&nbsp;&part;E/&part;Wf</strong></h1>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Same as the other posts, we will write down the basic equation and dimensionality first as a reference</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Dimensionality: M is defined as a hidden unit number, N is the input element number, T is the output element number</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As stated before, the summation is just the row of the weight matrix multiplied by the column of the input vector. And it represents one hidden unit out of M.</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Just remember that each page has different symbols such as a1 that are dedicated toward the local page only and never apply them to a different page.</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
        {Ftgeneral}
        {itgeneral}
        {Otgeneral}
        {Cdtgeneral}
        {Ctgeneral}
        {htgeneral}
        {Ligeneral}
        {Eigeneral}
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

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The first two terms are commonly found in other derivation as well. Those are needed in order to compute the scaled output from hidden units. In the original LSTM paper, it only has one output so their T is assumed to be one. But here we assume the output is T.</p>
                
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                    {dEoverdG_j_final}
                    {dgoverdL_i_final}
                    {dgoverdL_i_finalcont1}

                    </MathJax>
                </MathJaxContext>
                
                <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The next part until the next sub-title will compute the following formula</p>
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                {dLioverdH}
                </MathJax>
                </MathJaxContext>
                <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; And the above formula will be expanded to</p>
             
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {dLoverdL1}
                        {dL1overdL2}
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
      {dFtoverdd1}
      {dsigmoid1}


    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; It should be noted that variable d above is different from the other page&#39;s variable d. In this instance, the d is the input equation for the sigmoid for the Ft</p>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The next part until the next sub-title will compute the following formula</p>
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {d1overdWf}
                    </MathJax>
                </MathJaxContext>
                <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; And the above formula will be expanded to</p>
             
<MathJaxContext config={config} version={3}>
    <MathJax inline>
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