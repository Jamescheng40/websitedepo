import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';
function LSTMdEoverdWi(){
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
      
      const Ft = '$$ {F_t = }\\sigma_s{(\\sum_{i=0}^M W_f \\otimes x_t + bias(optional))} = \\sigma_s{( W_f \\odot x_t + bias(optional))}{, \\; where \\; \\sigma_s \\; is \\; sigmoid}$$';
      const it = '$$ {i_t = }\\sigma_s{(\\sum_{i=0}^M W_i \\otimes x_t + bias(optional))} = \\sigma_s{( W_i \\odot x_t + bias(optional))} = \\sigma_s{(d)} \\; {where \\; d=\\sigma_s(\\sum_{i=0}^M W_i \\otimes x_t) \\; is \\; sigmoid \\; and \\; d1=\\sum_{i=0}^M d2}, \\; d2=W_i \\otimes x_t   $$';
      const Ot = '$$ {O_t = }\\sigma_s{(\\sum_{i=0}^M W_o \\otimes x_t + bias(optional))} = \\sigma_s{( W_o \\odot x_t + bias(optional))} $$';
      const Cdt = "$$ {C'_t = }\\sigma_t{(\\sum_{i=0}^M W_c \\otimes x_t + bias(optional))} = \\sigma_s{( W_c \\odot x_t + bias(optional))}{, \\; where \\; \\sigma_t \\; is \\; tanh} $$";
      const Ct =  "$$ {C_t = }  F_t \\otimes {(C_{t-1})} \\oplus i_t \\otimes C'_t {, \\; where \\; \\otimes \\; is \\; element-wise \\; multiply \\; and \\oplus \\; is \\; element-wise \\; addition  ,where \\; a=F_t \\otimes {(C_{t-1})} \\; and \\; a1= i_t \\otimes C'_t }$$";
      const ht  = "$$ {h_t = }  O_t \\otimes \\sigma_t{(C_t)} = {O_t \\otimes b_t} {, \\; where \\; b_t \\; is \\; \\sigma_t{(C_t)}} $$";
      const Li = "$$ {L_t =  W_v \\odot h_t =}{\\sum_{i=0}^T W_v \\otimes h_t + bias(optional)}, \\; where \\; L1 =\\sum_{i=0}^M L2,and \\; L2=W_v \\otimes h_t $$"
      const Pi = "$$ {P_i = softmax(L_i)}{=\\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}} } $$"
      const Ei = "$$ {E_i=}{- \\sum_{i=0}^T Y_i  \\otimes log(P_i) + bias(optional)}{= - \\sum_{i=0}^T g,where \\; g = Y_i  \\otimes log(P_i)} $$"
      const dEoverdW = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_i} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_i} \\oplus \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_i} ) \\oplus \\frac{\\partial C_t}{\\partial a1} \\odot (\\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot (\\frac{\\partial d7}{\\partial W_i} \\oplus \\frac{\\partial d7}{\\partial x_t} \\odot \\frac{\\partial x_t}{\\partial W_i} ) \\oplus \\frac{\\partial a1}{\\partial C'_t} \\odot \\frac{\\partial C'_t}{\\partial W_i} ) )))}    $$";
      const dEoverdWcrossout = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_i} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_i}} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot ( \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_i} \\oplus \\cancelto{0}{\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_i}} ) \\oplus \\frac{\\partial C_t}{\\partial a1} \\odot (\\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot (\\frac{\\partial d7}{\\partial W_i} \\oplus \\cancelto{0}{\\frac{\\partial d7}{\\partial x_t} \\odot \\frac{\\partial x_t}{\\partial W_i}} ) \\oplus \\cancelto{0}{\\frac{\\partial a1}{\\partial C'_t} \\odot \\frac{\\partial C'_t}{\\partial W_i}} ) ))) }   $$";
      const dEoverdwsimplified =" $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_i} \\oplus  \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot \\frac{\\partial d7}{\\partial W_i} ) $$"
      const dEoverdwicomputefirst = "$$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot \\frac{\\partial d7}{\\partial W_i} $$"

      const dEoverdG_j_final = "$$ \\frac{\\partial E}{\\partial g_j} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times T   $$"      
      const dgoverdL_i_final = "$$ \\frac{\\partial g_j}{\\partial L_t} = \\frac{\\partial \\begin{bmatrix} (L_1 - Y_1)^2 \\\\ (L_2 - Y_2)^2 \\\\ \\vdots \\\\ (L_T - Y_T)^2 \\end{bmatrix}}{\\partial L_t} = \\begin{bmatrix} \\frac{\\partial (L_1 - Y_1)^2}{\\partial L_1} & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_1 - Y_1)^2}{\\partial L_T}} \\\\ \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_1}} & \\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_T}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}} & \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_2}} & \\cdots & \\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}  \\end{bmatrix} = diag(2 \\times (L_1 - Y_1)) \\ni T \\times T  $$"
      const dgoverdL_i_finalcont1 = "$$ \\frac{\\partial (L_1 - Y_1)^2}{\\partial L_1} = 2 \\times (L_1 - Y_1) \\times (\\cancelto{1}{\\frac{\\partial L_1}{\\partial L_1}} - \\cancelto{0}{\\frac{\\partial Y_1}{\\partial L_1}})   $$"
      
      const dEoverdG_j = "$$ \\frac{\\partial E}{\\partial g_j} = - \\frac{\\partial \\sum_{i=0}^T g_i}{\\partial g_j} =  \\begin{bmatrix} \\frac{\\partial (g_1 + g_2 + \\cdots + g_T)}{\\partial g_1} & \\frac{\\partial (g_1 + g_2 + \\cdots + g_T)}{\\partial g_2} \\cdots \\frac{\\partial (g_1 + g_2 + \\cdots + g_T)}{\\partial g_T} \\end{bmatrix} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times T  $$"
      const dgoverdP_i = "$$\\frac{\\partial g}{\\partial P_i} = \\frac{\\partial \\begin{bmatrix} Y_1 \\times \\log(P_1) \\\\ Y_2 \\times \\log(P_2) \\\\ \\vdots \\\\ Y_T \\times \\log(P_T) \\end{bmatrix}}{\\partial P_i} = \\begin{bmatrix} \\frac{\\partial (Y_1 \\times \\log(P_1))}{\\partial P_1} & \\cancelto{0}{\\frac{\\partial (Y_1 \\times \\log(P_1))}{\\partial P_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (Y_1 \\times \\log(P_1))}{\\partial P_T}} \\\\ \\cancelto{0}{\\frac{\\partial (Y_2 \\times \\log(P_2))}{\\partial P_1}} & \\frac{\\partial (Y_2 \\times \\log(P_2))}{\\partial P_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (Y_2 \\times \\log(P_2))}{\\partial P_T}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (Y_T \\times \\log(P_T))}{\\partial P_1}} & \\cancelto{0}{\\frac{\\partial (Y_T \\times \\log(P_T))}{\\partial P_2}} & \\cdots & \\frac{\\partial (Y_T \\times \\log(P_T))}{\\partial P_T}  \\end{bmatrix} = diag(Y_i/P_i) \\ni T \\times T   $$"
      const test12341 = "$$ \\frac{\\partial(Y_1 \\times \\log(P_1))}{\\partial P_1} = \\frac{Y_1}{P_1}, since \\; derivative \\; of \\; log(x) = \\frac{1}{x} $$ "
      const test12342 = "$$ \\frac{\\partial(Y_1 \\times \\log(P_1))}{\\partial P_2} = 0 $$"
      const quotientruleformula = "$$ \\frac{d}{dx} \[ \\frac{f(x)}{z(x)} \] = \\frac{z(x)f'(x)-f(x)z'(x) }{(z(x)^2)} $$"
      const applyingquotientrule1 = "$$ f(x)=e^{L_1}   $$"
      const applyingquotientrule2 = "$$ z(x)=\\sum_{i=0}^T e^{L_k} $$"
      const applyingquotientrule3 = "$$ f'(x) = \\frac{\\partial e^{L_1}}{\\partial L_1} = e^{L_1} \\times \\frac{\\partial e^{L_1}}{\\partial L_1} = e^{L_1} \\times 1  $$"
      const applyingquotientrule4 = "$$ z'(x) = \\frac{\\partial \\sum_{i=0}^T e^{L_k}}{\\partial L_1} = e^{L_1}    $$"
      const applyingquotientrule5 = "$$ \\frac{\\partial  (\\frac{e^{L_1}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_1}}=\\frac{z(x)f'(x)-f(x)z'(x)}{(z(x)^2)}=\\frac{ \\sum_{i=0}^T e^{L_k} \\times e^{L_1} - e^{L_1} \\times e^{L_1}  }{(\\sum_{i=0}^T e^{L_k})^2} = \\frac{e^{L_1}}{\\sum_{i=0}^T e^{L_k}} - \\frac{(e^{L_1})^2}{\\sum_{i=0}^T e^{L_k}} = P_1 \\times (1 - P_1) $$"
      const applyingquotientrule6 = "Because $$  P_i(L_i) = \\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}}  $$"
      const applyingquotientrule7 = "$$   $$"   
      const nondiagnal1 = "$$ f(x)=e^{L_1}   $$"
      const nondiagnal2 = "$$ z(x)=\\sum_{i=0}^T e^{L_k} $$"
      const nondiagnal3 = "$$ f'(x) = \\frac{\\partial e^{L_1}}{\\partial L_2} = 0  $$"
      const nondiagnal4 = "$$ z'(x) = \\frac{\\partial \\sum_{i=0}^T e^{L_k}}{\\partial L_2} = e^{L_2}    $$"
      const nondiagnal5 = "$$ \\frac{\\partial  (\\frac{e^{L_1}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_2}}=\\frac{z(x)f'(x)-f(x)z'(x)}{(z(x)^2)}=\\frac{ \\sum_{i=0}^T e^{L_k} \\times 0 - e^{L_1} \\times e^{L_2}  }{(\\sum_{i=0}^T e^{L_k})^2} = 0 - \\frac{e^{L_1} \\times e^{L_2}}{(\\sum_{i=0}^T e^{L_k})^2} =  - P_1 \\times P_2 $$"
      const nondiagnal6 = "Because $$  P_i(L_i) = \\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}}  $$"
      const dpOverdLi = "$$\\frac{\\partial P_i}{\\partial L_i}=\\begin{bmatrix} \\frac{\\partial  (\\frac{e^{L_1}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_1}} & \\frac{\\partial (\\frac{e^{L_1}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_2}} & \\cdots & \\frac{\\partial (\\frac{e^{L_1}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_T}} \\\\ \\frac{\\partial (\\frac{e^{L_2}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_1}} & \\frac{\\partial (\\frac{e^{L_2}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_2}} & \\cdots & \\frac{\\partial (\\frac{e^{L_2}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_T}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\frac{\\partial (\\frac{e^{L_T}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_1}} & \\frac{\\partial (\\frac{e^{L_T}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_2}} & \\cdots & \\frac{\\partial (\\frac{e^{L_T}}{e^{L_1}+e^{L_2} + \\cdots + e^{L_T}})}{\\partial e^{L_T}}  \\end{bmatrix}   =   \\begin{bmatrix} P_1 \\times (1 - P_1) & - P_1 \\times P_2 & \\cdots & -P_1 \\times P_T \\\\ -P_2 \\times P_1 & P_2 \\times (1 - P_2) & \\cdots & P_2 \\times P_T  \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ -P_T \\times P_1 & -P_T \\times P_2 & \\cdots & P_T \\times (1 - P_T) \\end{bmatrix} \\ni T \\times T $$ "
      const dLoverdL1 = "$$ \\frac{\\partial L_i}{\\partial L1_{1}} = 1 $$"
      //const equation = '    $$\\left[ \\begin{array}{cc|c} 1&2&3\\\\ 4&5&6  \\end{array} \\right] $$ ';
      //const summation = '$$\\sigma_s{(\\sum_{i=0}^M W_f \\otimes x_t + bias(optional))} $$' 
      const dL1overdL2 = "$$ \\frac{\\partial L1_1}{\\partial L2_1} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M; \\; \\frac{\\partial L1_2}{\\partial L2_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M \\cdots \\frac{\\partial L1_T}{\\partial L2_T} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M $$"
      const dL2overdh = "$$ \\frac{\\partial L2_1}{\\partial h_t} = \\frac{\\partial \\begin{bmatrix} W_{v11} \\times h_1 \\\\ W_{v12} \\times h_2 \\\\ \\vdots \\\\ W_{v1M} \\times h_M \\end{bmatrix}}{\\partial h_t} = \\begin{bmatrix} \\frac{\\partial (W_{v11} \\times h_1)}{\\partial h_1} & \\cancelto{0}{\\frac{\\partial (W_{v11} \\times h_1)}{\\partial h_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{v11} \\times h_1)}{\\partial h_M}} \\\\ \\cancelto{0}{\\frac{\\partial (W_{v12} \\times h_2)}{\\partial h_1}} & \\frac{\\partial (W_{v12} \\times h_2)}{\\partial h_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{v12} \\times h_2)}{\\partial h_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{v1M} \\times h_M)}{\\partial h_1}} & \\cancelto{0}{\\frac{\\partial (W_{v1M} \\times h_M)}{\\partial h_2}} & \\cdots & \\frac{\\partial (W_{v1M} \\times h_M)}{\\partial h_M}  \\end{bmatrix} = \\begin{bmatrix} W_{v11} & 0 & \\cdots & 0 \\\\ 0 & W_{v12}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{v1M} \\end{bmatrix} = diag(W_{v1i}) \\ni M \\times M   $$"
      
      const dL22overdh2 = "$$\\frac{\\partial L2_2}{\\partial h_t} = \\frac{\\partial \\begin{bmatrix} W_{v21} \\times h_1 \\\\ W_{v22} \\times h_2 \\\\ \\vdots \\\\ W_{v2M} \\times h_M \\end{bmatrix}}{\\partial h_t} = \\begin{bmatrix} \\frac{\\partial (W_{v21} \\times h_1)}{\\partial h_1} & \\cancelto{0}{\\frac{\\partial (W_{v21} \\times h_1)}{\\partial h_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{v21} \\times h_1)}{\\partial h_M}} \\\\ \\cancelto{0}{\\frac{\\partial (W_{v22} \\times h_2)}{\\partial h_1}} & \\frac{\\partial (W_{v22} \\times h_2)}{\\partial h_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{v22} \\times h_2)}{\\partial h_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{v2M} \\times h_M)}{\\partial h_1}} & \\cancelto{0}{\\frac{\\partial (W_{v2M} \\times h_M)}{\\partial h_2}} & \\cdots & \\frac{\\partial (W_{v2M} \\times h_M)}{\\partial h_M}  \\end{bmatrix} = \\begin{bmatrix} W_{v21} & 0 & \\cdots & 0 \\\\ 0 & W_{v22}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{v2M} \\end{bmatrix} = diag(W_{v2i}) \\ni M \\times M   $$"
      
      const dL21overdh2timesdL2overdh = "$$  \\frac{\\partial L1_1}{\\partial L2_1} \\odot \\frac{\\partial L2_1}{\\partial h_t} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} W_{v11} & 0 & \\cdots & 0 \\\\ 0 & W_{v12}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{v1M} \\end{bmatrix} = \\begin{bmatrix} W_{v11} & W_{v12} & \\cdots & W_{v1M}  \\end{bmatrix} $$"
      
      const dL22overdh2timesdL2overdh = "$$  \\frac{\\partial L1_2}{\\partial L2_2} \\odot \\frac{\\partial L2_2}{\\partial h_t}  = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} W_{v21} & 0 & \\cdots & 0 \\\\ 0 & W_{v22}  & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & W_{v2M} \\end{bmatrix} =  \\begin{bmatrix} W_{v21} & W_{v22} & \\cdots & W_{v2M}  \\end{bmatrix} $$"
           
      const dL2ioverdh2i = "$$ \\frac{\\partial Li}{\\partial h_t} = \\begin{bmatrix} W_{v11} & W_{v12} & \\cdots & W_{v1M} \\\\ W_{v21} & W_{v22} & \\cdots & W_{v2M} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{vT1} & W_{vT2} & \\cdots & W_{vTM} \\end{bmatrix} \\ni T \\times M $$"
      
      const dhtoverdb = "$$\\frac{\\partial h_t}{\\partial b_t} = \\frac{\\partial \\begin{bmatrix} O_{{t}_{1}} \\times b_{{t}_{1}} \\\\ O_{{t}_{2}} \\times b_{{t}_{2}} \\\\ \\vdots \\\\ O_{{t}_{M}} \\times b_{{t}_{M}} \\end{bmatrix}}{\\partial b_i} = \\begin{bmatrix} \\frac{\\partial (O_{{t}_{1}} \\times b_{{t}_{1}})}{\\partial b_{{t}_{1}}} & \\cancelto{0}{\\frac{\\partial (O_{{t}_{1}} \\times b_{{t}_{1}})}{\\partial b_{{t}_{2}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (O_{{t}_{1}} \\times b_{{t}_{1}})}{\\partial b_{{t}_{M}}}} \\\\ \\cancelto{0}{\\frac{\\partial (O_{{t}_{2}} \\times b_{{t}_{2}})}{\\partial b_{{t}_{1}}}} & \\frac{\\partial (O_{{t}_{2}} \\times b_{{t}_{2}})}{\\partial b_{{t}_{2}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (O_{{t}_{2}} \\times b_{{t}_{2}})}{\\partial b_{{t}_{M}}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (O_{{t}_{M}} \\times b_{{t}_{M}})}{\\partial b_{{t}_{1}}}} & \\cancelto{0}{\\frac{\\partial (O_{{t}_{M}} \\times b_{{t}_{M}})}{\\partial b_{{t}_{2}}}} & \\cdots & \\frac{\\partial (O_{{t}_{M}} \\times b_{{t}_{M}})}{\\partial b_{{t}_{M}}}  \\end{bmatrix} = diag(O_t) \\ni M \\times M $$"
      
      const dboverdCt = "$$\\frac{\\partial b_t}{\\partial C_t} = \\frac{\\partial \\begin{bmatrix} \\tanh(C_1) \\\\ \\tanh(C_2) \\\\ \\vdots \\\\ \\tanh(C_M) \\end{bmatrix}}{\\partial C_t} = \\begin{bmatrix} \\frac{\\partial (\\tanh(C_1))}{\\partial C_1} & \\cancelto{0}{\\frac{\\partial (\\tanh(C_1))}{\\partial C_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (\\tanh(C_1))}{\\partial C_M}} \\\\ \\cancelto{0}{\\frac{\\partial (\\tanh(C_2))}{\\partial C_1}} & \\frac{\\partial (\\tanh(C_2))}{\\partial C_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (\\tanh(C_2))}{\\partial C_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (\\tanh(C_M))}{\\partial C_1}} & \\cancelto{0}{\\frac{\\partial (\\tanh(C_M))}{\\partial C_2}} & \\cdots & \\frac{\\partial (tanh(C_M))}{\\partial C_M}  \\end{bmatrix} = 1 - \\tanh^2(C_t) \\ni M \\times M $$"
      const dtanh = "$$ \\frac{\\partial \\tanh(C_1)}{\\partial C_1} = (1 - \\tanh^2(C_1)) $$"
      
      const dCtoverda1 = "$$ \\frac{\\partial C_t}{\\partial a1} = \\frac{\\partial (a + a1)}{\\partial a1} = 1 $$" 
      const da1overdit = "$$ \\frac{\\partial a1_t}{\\partial i_t} = \\frac{\\partial \\begin{bmatrix} i_1 \\times C'_1 \\\\ i_2 \\times C'_2 \\\\ \\vdots \\\\ i_M \\times C'_M \\end{bmatrix}}{\\partial i_t} = \\begin{bmatrix} \\frac{\\partial (i_1 \\times C'_1)}{\\partial i_1} & \\cancelto{0}{\\frac{\\partial (i_1 \\times C'_1)}{\\partial i_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (i_1 \\times C'_1)}{\\partial i_M}} \\\\ \\cancelto{0}{\\frac{\\partial (i_2 \\times C'_2)}{\\partial i_1}} & \\frac{\\partial (i_2 \\times C'_2)}{\\partial i_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (i_2 \\times C'_2)}{\\partial i_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (i_M \\times C'_M)}{\\partial i_M}} & \\cancelto{0}{\\frac{\\partial (i_M \\times C'_M)}{\\partial i_2}} & \\cdots & \\frac{\\partial (i_M \\times C'_M)}{\\partial i_M}  \\end{bmatrix} = diag(C'_t) \\ni M \\times M $$"
      
      const ditoverdd = "$$\\frac{\\partial i_t}{\\partial d5_t} = \\frac{\\partial \\begin{bmatrix} sigmoid(d5_1) \\\\ sigmoid(d5_2) \\\\ \\vdots \\\\ sigmoid(d5_M) \\end{bmatrix}}{\\partial d5_t} = \\begin{bmatrix} \\frac{\\partial (sigmoid(d5_1))}{\\partial d5_1} & \\cancelto{0}{\\frac{\\partial (sigmoid(d5_1))}{\\partial d5_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (sigmoid(d5_1))}{\\partial d5_M}} \\\\ \\cancelto{0}{\\frac{\\partial (sigmoid(d5_2))}{\\partial d5_1}} & \\frac{\\partial (sigmoid(d5_2))}{\\partial d5_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (sigmoid(d5_2))}{\\partial d5_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (sigmoid(d5_M))}{\\partial d5_1}} & \\cancelto{0}{\\frac{\\partial (sigmoid(d5_M))}{\\partial d5_2}} & \\cdots & \\frac{\\partial (sigmoid(d5_M))}{\\partial d5_M}  \\end{bmatrix} = sigmoid(d5_t) \\times (1 - sigmoid(d5_t)) \\ni M \\times M $$"
      const dsigmoid = "$$ \\frac{\\partial (sigmoid(d5_1))}{\\partial d5_1} = sigmoid(d5_1) \\times (1 - sigmoid(d5_1)) $$"
      
      
      const ddoverd1 = "$$  \\frac{\\partial d6_1}{\\partial d7_1} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N; \\; \\frac{\\partial d6_2}{\\partial d7_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N \\cdots \\frac{\\partial d6_T}{\\partial d7_T} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N $$"
      
      const d21overdWi = "$$ \\frac{\\partial d7_1}{\\partial Wi} = \\frac{\\partial \\begin{bmatrix} W_{i11} \\times x_1 \\\\ W_{i12} \\times x_2 \\\\ \\vdots \\\\ W_{i1N} \\times x_N \\end{bmatrix}}{\\partial W_i} = \\begin{bmatrix} \\frac{\\partial ( W_{i11} \\times x_1 )}{\\partial W_{i11}} & \\cancelto{0}{\\frac{\\partial (W_{i11} \\times x_1)}{\\partial W_{i12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{i11} \\times x_1)}{\\partial W_{i1N}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{i11} \\times x_1)}{\\partial W_{iMN}}} & \\cdots & 0 \\\\ \\cancelto{0}{\\frac{\\partial ((W_{i12} \\times x_2))}{\\partial W_{i11}}} & \\frac{\\partial ((W_{i12} \\times x_2))}{\\partial W_{i12}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{i12} \\times x_2))}{\\partial W_{i1N}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{i12} \\times x_2))}{\\partial W_{iMN}}} & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{i1N} \\times x_N)}{\\partial W_{i11}}} & \\cancelto{0}{\\frac{\\partial (W_{i1N} \\times x_N)}{\\partial W_{i12}}} & \\cdots & \\frac{\\partial (W_{i1N} \\times x_N)}{\\partial W_{i1N}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{i1N} \\times x_N)}{\\partial W_{iMN}}} & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix}  \\ni N \\times MN $$ "
      const d22overdWi = "$$ \\frac{\\partial d7_2}{\\partial Wi} = \\frac{\\partial \\begin{bmatrix} W_{i11} \\times x_1 \\\\ W_{i12} \\times x_2 \\\\ \\vdots \\\\ W_{i1N} \\times x_N \\end{bmatrix}}{\\partial W_i} = \\begin{bmatrix} \\cancelto{0}{\\frac{\\partial (W_{i21} \\times x_1)}{\\partial W_{i11}}} & \\cancelto{0}{\\frac{\\partial (W_{i21} \\times x_1)}{\\partial W_{i12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{i21} \\times x_1)}{\\partial W_{i1N}}} & \\frac{\\partial ( W_{i21} \\times x_1 )}{\\partial W_{i21}} & \\cancelto{0}{\\frac{\\partial (W_{i21} \\times x_1)}{\\partial W_{i22}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{i21} \\times x_1)}{\\partial W_{i2N}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{i21} \\times x_1)}{\\partial W_{iMN}}}  \\\\ \\cancelto{0}{\\frac{\\partial ((W_{i22} \\times x_2))}{\\partial W_{i11}}} & \\cancelto{0}{\\frac{\\partial ((W_{i22} \\times x_2))}{\\partial W_{i12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{i22} \\times x_2))}{\\partial W_{i1N}}} & \\cancelto{0}{\\frac{\\partial ((W_{i22} \\times x_2))}{\\partial W_{i21}}} & \\frac{\\partial ((W_{i22} \\times x_2))}{\\partial W_{i22}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{i12} \\times x_2))}{\\partial W_{i2N}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{i12} \\times x_2))}{\\partial W_{iMN}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{i2N} \\times x_N)}{\\partial W_{i11}}} & \\cancelto{0}{\\frac{\\partial (W_{i2N} \\times x_N)}{\\partial W_{i12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{i2N} \\times x_N)}{\\partial W_{i1N}}} & \\cancelto{0}{\\frac{\\partial (W_{i2N} \\times x_N)}{\\partial W_{i21}}} & \\cancelto{0}{\\frac{\\partial (W_{i2N} \\times x_N)}{\\partial W_{i22}}} & 0 & \\frac{\\partial (W_{i2N} \\times x_N)}{\\partial W_{i2N}} & \\cdots &  \\cancelto{0}{\\frac{\\partial (W_{i2N} \\times x_N)}{\\partial W_{iMN}}}  \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix}  \\ni N \\times MN $$ "
      
      const ddoverd1timesd21overdWi = "$$ \\frac{\\partial d6_1}{\\partial d7_1} \\odot \\frac{\\partial d7_1}{\\partial Wi} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
      const ddoverd12timesd21overdWi = "$$ \\frac{\\partial d6_2}{\\partial d7_2} \\odot \\frac{\\partial d7_2}{\\partial Wi} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
      const ddoverd1timesd21overdWicombine = "$$ \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} \\ni M \\times MN  $$"
      
      const tequal3compute = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot (\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}} \\odot (\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}} \\odot(\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-3}} \\oplus \\frac{\\partial C_{t-3}}{\\partial a1_{t-3}} \\odot \\frac{\\partial a1_{t-3}}{\\partial i_{t-3}} \\odot \\frac{\\partial i_{t-3}}{\\partial d6_{t-3}} \\odot \\frac{\\partial d6_{t-3}}{\\partial d7_{t-3}} \\odot \\frac{\\partial d7_{t-3}}{\\partial W_{i_{t-3}}}) \\oplus \\frac{\\partial C_{t-2}}{\\partial a1_{t-2}} \\odot \\frac{\\partial a1_{t-2}}{\\partial i_{t-2}} \\odot \\frac{\\partial i_{t-2}}{\\partial d6_{t-2}} \\odot \\frac{\\partial d6_{t-2}}{\\partial d7_{t-2}} \\odot \\frac{\\partial d7_{t-2}}{\\partial W_{i_{t-2}}} )\\oplus \\frac{\\partial C_{t-1}}{\\partial a1_{t-1}} \\odot \\frac{\\partial a1_{t-1}}{\\partial i_{t-1}} \\odot \\frac{\\partial i_{t-1}}{\\partial d6_{t-1}} \\odot \\frac{\\partial d6_{t-1}}{\\partial d7_{t-1}} \\odot \\frac{\\partial d7_{t-1}}{\\partial W_{i_{t-1}}} ) \\oplus  \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot \\frac{\\partial d7}{\\partial W_i} ) $$"
      const tequal3computehighlighted = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\underline{ \\textcolor{blue}{\\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot(\\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus \\frac{\\partial C_{t-3}}{\\partial a1_{t-3}} \\odot \\frac{\\partial a1_{t-3}}{\\partial i_{t-3}} \\odot \\frac{\\partial i_{t-3}}{\\partial d6_{t-3}} \\odot \\frac{\\partial d6_{t-3}}{\\partial d7_{t-3}} \\odot \\frac{\\partial d7_{t-3}}{\\partial W_{i_{t-3}}}) \\oplus \\frac{\\partial C_{t-2}}{\\partial a1_{t-2}} \\odot \\frac{\\partial a1_{t-2}}{\\partial i_{t-2}} \\odot \\frac{\\partial i_{t-2}}{\\partial d6_{t-2}} \\odot \\frac{\\partial d6_{t-2}}{\\partial d7_{t-2}} \\odot \\frac{\\partial d7_{t-2}}{\\partial W_{i_{t-2}}} )\\oplus \\frac{\\partial C_{t-1}}{\\partial a1_{t-1}} \\odot \\frac{\\partial a1_{t-1}}{\\partial i_{t-1}} \\odot \\frac{\\partial i_{t-1}}{\\partial d6_{t-1}} \\odot \\frac{\\partial d6_{t-1}}{\\partial d7_{t-1}} \\odot \\frac{\\partial d7_{t-1}}{\\partial W_{i_{t-1}}} ) \\oplus  \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial i_t} \\odot \\frac{\\partial i_t}{\\partial d6} \\odot \\frac{\\partial d6}{\\partial d7} \\odot \\frac{\\partial d7}{\\partial W_i} ) $$"
      
      const dCtoverda = "$$ \\frac{\\partial C_t}{\\partial a_t}  =  \\frac{\\partial \\begin{bmatrix} a_1 + a1_1 \\\\ a_2 + a1_2 \\\\ \\vdots \\\\ a_M + a1_M \\end{bmatrix}}{\\partial a_t} = \\begin{bmatrix} \\frac{\\partial (a_1 + a1_1)}{\\partial a_1} & \\cancelto{0}{\\frac{\\partial (a_2 + a1_2)}{\\partial a_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (a_1 \\times a1_1)}{\\partial a_M}} \\\\ \\cancelto{0}{\\frac{\\partial (a_2 + a1_2)}{\\partial a_1}} & \\frac{\\partial (a_2 \\times a1_2)}{\\partial a_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (a_2 \\times a1_2)}{\\partial a_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (a_M \\times a1_M)}{\\partial a_M}} & \\cancelto{0}{\\frac{\\partial (a_M \\times a1_M)}{\\partial a_2}} & \\cdots & \\frac{\\partial (a_M \\times a1_M)}{\\partial a_M}  \\end{bmatrix} = diag(1) \\ni M \\times M $$"
      const daoverCtm1 = "$$ \\frac{\\partial a_t}{\\partial C_{t-1}} = \\frac{\\partial \\begin{bmatrix} F_{t_1} \\times C_{{t-1}_1} \\\\ F_{t_2} \\times C_{{t-1}_2} \\\\ \\vdots \\\\ F_{t_M} \\times C_{{t-1}_M} \\end{bmatrix}}{\\partial C_{t-1}} = \\begin{bmatrix} \\frac{\\partial (F_{t_1} \\times C_{{t-1}_1})}{\\partial C_{{t-1}_1} } & \\cancelto{0}{\\frac{\\partial (F_{t_1} \\times C_{{t-1}_1})}{\\partial C_{{t-1}_2}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (F_{t_M} \\times C_{{t-1}_M})}{\\partial C_{{t-1}_M}}} \\\\ \\cancelto{0}{\\frac{\\partial (F_{t_2} \\times C_{{t-1}_2})}{\\partial C_{{t-1}_1}}} & \\frac{\\partial (F_{t_2} \\times C_{{t-1}_2})}{\\partial C_{{t-1}_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (F_{t_2} \\times C_{{t-1}_2})}{\\partial C_{{t-1}_M}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (F_{t_M} \\times C_{{t-1}_M})}{\\partial C_{{t-1}_1}}} & \\cancelto{0}{\\frac{\\partial (F_{t_M} \\times C_{{t-1}_M})}{\\partial C_{{t-1}_2}}} & \\cdots & \\frac{\\partial (F_{t_M} \\times C_{{t-1}_M})}{\\partial C_{{t-1}_M}}  \\end{bmatrix} = diag(F_t) \\ni M \\times M $$"
      const datm1overCtm2 = "$$ \\frac{\\partial a_{t-1}}{\\partial C_{t-2}} = \\frac{\\partial \\begin{bmatrix} F_{{t-1}_1} \\times C_{{t-2}_1} \\\\ F_{{t-1}_2} \\times C_{{t-2}_2} \\\\ \\vdots \\\\ F_{{t-1}_M} \\times C_{{t-2}_M} \\end{bmatrix}}{\\partial C_{t-2}} = \\begin{bmatrix} \\frac{\\partial (F_{{t-1}_1} \\times C_{{t-2}_1})}{\\partial C_{{t-2}_1} } & \\cancelto{0}{\\frac{\\partial (F_{{t-1}_1} \\times C_{{t-2}_1})}{\\partial C_{{t-2}_2}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (F_{{t-1}_M} \\times C_{{t-2}_M})}{\\partial C_{{t-2}_M}}} \\\\ \\cancelto{0}{\\frac{\\partial (F_{{t-1}_2} \\times C_{{t-2}_2})}{\\partial C_{{t-2}_1}}} & \\frac{\\partial (F_{{t-1}_2} \\times C_{{t-2}_2})}{\\partial C_{{t-2}_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (F_{{t-1}_M} \\times C_{{t-2}_M})}{\\partial C_{{t-2}_M}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (F_{{t-1}_M} \\times C_{{t-2}_M})}{\\partial C_{{t-2}_1}}} & \\cancelto{0}{\\frac{\\partial (F_{{t-1}_M} \\times C_{{t-2}_M})}{\\partial C_{{t-2}_2}}} & \\cdots & \\frac{\\partial (F_{{t-1}_M} \\times C_{{t-2}_M})}{\\partial C_{{t-2}_M}}  \\end{bmatrix} = diag(F_{t-1}) \\ni M \\times M $$"
      const dtmioverdCtmim1 = "$$ \\frac{\\partial a_{t-i}}{\\partial C_{t-i-1}} = diag(F_{t-i})  $$"
            return (
        
        <div>

<h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<strong>FULL&nbsp;</strong><span style={{fontSize:37.379999999999995}}><strong>Derivation of&nbsp;</strong></span><strong>&nbsp;&part;E/&part;Wᵢ</strong></h1>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;The Related equation can be listed below or you can refer to the main page summary for the equation:</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Dimensionality: M is defined as a hidden unit number, N is the input element number, T is the output element number</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;In the below function, you might find that the dot product&nbsp;<strong>⨀&nbsp;</strong>is replaced with&nbsp;<strong>&sum;a &otimes; b, a detailed explanation of why can be found <a href="http://a">here</a>&nbsp;</strong></p>

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




<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As you can see from above, the&nbsp;<strong>Wᵢ&nbsp;</strong>is related to several other components in the equations from <strong>hₜ</strong><span style={{fontSize:24.03}}><strong>&nbsp;</strong></span>down to <strong>Fᵢ</strong>. Specifically, we can use one notation to summarize the dependency,&nbsp;<strong>&part;E(Pi, Lᵢ,&nbsp;hₜ, Cₜ, Fᵢ, Fcₜ₋₁)/&part;Wᵢ</strong></p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;With the above information, it would be enough to write down the entire equation. So let us write down the entire equation:</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                {dEoverdW}
                </MathJax>
            </MathJaxContext>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Don&#39;t be confused and scared by the convoluted equation above, some of the terms will become 0 and can be ignored safely. Let us continue to simplify the equation</p>

<MathJaxContext config={config} version={3}>
                <MathJax inline>
                {dEoverdWcrossout}
                </MathJax>
            </MathJaxContext>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style={{fontSize:24.03}}><strong>&part;hₜ/&part;Fₒ&nbsp;⨀ &part;Fₒ/&part;Wᵢ = 0 </strong></span>since there is no Wi in&nbsp;Fₒ. It is the same for others, just look it up in the equations above.</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                {dEoverdwsimplified}
                </MathJax>
            </MathJaxContext>
<p><span style={{fontSize:24.03}}><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</strong></span>&nbsp;Notice above that the giant term before the element-wise&nbsp;<span style={{fontSize:24.03}}><strong>&oplus;&nbsp;</strong></span>could be expanded into several terms depending on the time steps we have. For example, t = 4 could be expanded into 3 terms. And the giant term on the right is a constant, even if the time step only has 1 or t = 1. We will have an example for <strong><u>timestep = 1</u></strong> implementation demonstration below. For timestep = 1, the formula is</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                {dEoverdwicomputefirst}
                </MathJax>
            </MathJaxContext>
<p><span style={{fontSize:24.03}}><strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</strong></span>OK now we have the equation expanded, we will have to turn each derivative within the term into a matrix.&nbsp;</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Let us start with the first three terms</p>
<MathJaxContext config={config} version={3}>
                    <MathJax inline>
                    {dEoverdG_j_final}
                    {dgoverdL_i_final}
                    {dgoverdL_i_finalcont1}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Since the below subscript of L1 is the row number of the TxT scalar</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Recall that&nbsp;</p>
<MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dLoverdL1}
                    {dL1overdL2}
                    {dL2overdh}
                    {dL22overdh2}
                    {dL21overdh2timesdL2overdh}
                    {dL22overdh2timesdL2overdh}
                    
                </MathJax>
            </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; so after the above deduction, we can conclude that&nbsp; <strong>&part;L2ᵢ/&part;hₜ&nbsp; &nbsp; </strong>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</p>
<MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dL2ioverdh2i}
                </MathJax>
            </MathJaxContext>

<p><span style={{fontSize:24.03}}><strong>&nbsp; &nbsp; &nbsp; &nbsp;</strong></span>Notice the dimension of Wv is TXM because we can T output and M is the vector size of the hidden layer</p>

<MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dhtoverdb}
                    {dboverdCt}
                    {dtanh}
                    {dCtoverda1}
                    {da1overdit}
                    {ditoverdd}
                    {dsigmoid}
                    
                </MathJax>
            </MathJaxContext>





<p>Notice the dimension of Wv is TXM because we can T output and M is the vector size of the hidden layer</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {ddoverd1}
                    {d21overdWi}
                    {d22overdWi}
                    {ddoverd1timesd21overdWi}
                    {ddoverd12timesd21overdWi}
                    {ddoverd1timesd21overdWicombine}
                    
                </MathJax>
            </MathJaxContext>

<p>Above is all for the requirement we need to compute the first time step, let us try to compute the for some longer timestep in order to find the pattern within it</p>

<p>Let us try a timestep = 3 and see how backpropagation could expand</p>
            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dEoverdwsimplified}
                    {tequal3compute}

                </MathJax>
            </MathJaxContext>

<p><u>Notice there is a term</u> crossed out above because there won&#39;t be a 4th timestep in this case. So let us expand this out:</p>

            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {tequal3computehighlighted}
                </MathJax>
            </MathJaxContext>

<p>The Green highlighted above shows the new computation term for each new backpropagation time step. Notice that there is only one new term added for each time step we go back so let us just compute those and we should be good because all other terms have already been computed from the previous step. The terms highlighted in Purple are the terms computed in the previous time step and should have already been saved.&nbsp; So let us just compute the green terms, only the first two terms would be</p>

<MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dCtoverda}
                    {daoverCtm1}
                    {datm1overCtm2}
                </MathJax>
            </MathJaxContext>
            
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Following this logic, the new step can be deduced to&nbsp;</p>
            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dtmioverdCtmim1}
                </MathJax>
            </MathJaxContext>
        </div>
    )
}
  
export default LSTMdEoverdWi;