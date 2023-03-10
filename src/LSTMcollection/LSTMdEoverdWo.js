import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';    

function LSTMdEoverdWo(){
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
      //const Pigeneral = "$$ {P_i = softmax(L_i)}{=\\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}} } $$"
      const Eigeneral = "$$ {E_i=}{ \\frac{1}{n} \\otimes \\sum_{i=0}^T (L_t - Y)^2  }{= \\frac{1}{n} \\otimes \\sum_{i=0}^T g,where \\; g = (L_t - Y)^2}.\\; The \\; Loss \\; function\\; here\\; is\\; the\\; Mean \\; Squred \\; Error $$"
      const dEoverdWo = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_o} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_o} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_o} \\oplus \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_o} ) \\oplus \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_o}))}    $$";
      const dEoverdWocrossedout = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_o} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_o} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\cancelto{0}{\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_o}} \\oplus \\cancelto{0}{\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_o}} ) \\oplus \\cancelto{0}{\\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_o}}))}    $$";
             
      const dEoverdWoexpanded = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_o} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_o}}} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\cancelto{0}{\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_o}} \\oplus \\cancelto{0}{\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_o}} ) \\oplus \\cancelto{0}{\\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_o}}))}    $$";
      const dEoverdWosimplified = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_o} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t}  \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_o}} \\oplus { \\frac{\\partial h_t}{\\partial O_t}  \\odot \\frac{\\partial O_t}{\\partial d13} \\odot \\frac{\\partial d13}{\\partial d14} \\odot \\frac{\\partial d14}{\\partial W_o} }) }    $$";
      
      
      const dEoverdWosimplifiedcancled = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_o} {=} \\frac{\\partial E}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t}  \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_o}} \\oplus { \\frac{\\partial h_t}{\\partial O_t}  \\odot \\frac{\\partial O_t}{\\partial d13} \\odot \\frac{\\partial d13}{\\partial d14} \\odot \\frac{\\partial d14}{\\partial W_o} }) }    $$";
      
      const dEoverdWosimplifiedcancledcont = "$$ = \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot { \\frac{\\partial h_t}{\\partial O_t}  \\odot \\frac{\\partial O_t}{\\partial d13} \\odot \\frac{\\partial d13}{\\partial d14} \\odot \\frac{\\partial d14}{\\partial W_o} } $$"
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
      const dL2ioverdh2ifinalproduct = "$$ \\frac{\\partial Li}{\\partial h_t} = \\begin{bmatrix} W_{{v}_{11}} & W_{{v}_{12}} & \\cdots & W_{{v}_{1M}} \\\\ W_{{v}_{21}} & W_{{v}_{22}} & \\cdots & W_{{v}_{2M}} \\\\ \\vdots & \\vdots & \\vdots \\\\ W_{{v}_{T1}} & W_{{v}_{T2}} & \\cdots & W_{{v}_{TM}} \\end{bmatrix} \\ni T \\times M $$"
      
      const dhttoverdOttwo = "$$ \\frac{\\partial h_t}{\\partial O_t} =  \\frac{\\partial \\begin{bmatrix} O_{{t}_1} \\times b_{{t}_1} \\\\ O_{{t}_2} \\times b_{{t}_2} \\\\ \\vdots \\\\ O_{{t}_M} \\times b_{{t}_M} \\end{bmatrix}}{\\partial O_t} = diag(b_{t}) = diag(\\sigma_t(C_t)) \\ni M \\times M $$"
      const dOtoverd13wo = "$$ \\frac{\\partial O_t}{\\partial d13_t} = \\frac{\\partial \\begin{bmatrix} sigmoid(d13_{{t}_1}) \\\\ sigmoid(d13_{{t}_2}) \\\\ \\vdots \\\\ sigmoid(d13_{{t}_M}) \\end{bmatrix}}{\\partial d13_{{t}}} = diag(sigmoid(d13_t) \\times (1 - sigmoid(d13_t))) \\ni M \\times M $$"
      
      const dd13overd14wo = "$$ \\frac{\\partial d13_1}{\\partial d14_1} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N; \\; \\frac{\\partial d13_2}{\\partial d14_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N \\cdots \\frac{\\partial d13_N}{\\partial d14_N} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N $$"
      const d141overdwo = "$$ \\frac{\\partial d14_1}{\\partial W_o} = \\frac{\\partial \\begin{bmatrix} W_{{o}_{11}} \\times x_1 \\\\ W_{{o}_{12}} \\times x_2 \\\\ \\vdots \\\\ W_{{o}_{1N}} \\times x_N \\end{bmatrix}}{\\partial W_o}  = \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix}  =  \\ni N \\times MN $$ "
      const d142overdWo = "$$ \\frac{\\partial d14_2}{\\partial W_o} = \\frac{\\partial \\begin{bmatrix} W_{{o}_{21}} \\times x_1 \\\\ W_{{o}_{22}} \\times x_2 \\\\ \\vdots \\\\ W_{{o}_{2N}} \\times x_N \\end{bmatrix}}{\\partial W_o}  = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix}  =  \\ni N \\times MN $$ "
      
      const dd13overd14t1imesd14overdWo = "$$ \\frac{\\partial d13_1}{\\partial d14_1} \\odot \\frac{\\partial d14_1}{\\partial W_o} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
      const dd13overd14timesd14overdWo = "$$ \\frac{\\partial d13_2}{\\partial d14_2} \\odot \\frac{\\partial d14_2}{\\partial W_o} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
      const ddoverd1timesd21overdWocombine1 = "$$ \\frac{\\partial d13}{\\partial W_o} = \\frac{\\partial d13}{\\partial d14} \\odot \\frac{\\partial d14}{\\partial W_o} = \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} \\ni M \\times MN  $$"
      
      const tequal3computenohighlightedwo = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot  ( \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot ( \\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}} \\odot ( {\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}} \\odot(\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}} \\oplus { \\frac{\\partial h_{t-3}}{\\partial O_{t-3}}  \\odot \\frac{\\partial O_{t-3}}{\\partial d13_{t-3}} \\odot \\frac{\\partial d13_{t-3}}{\\partial d14_{t-3}} \\odot \\frac{\\partial d14_{t-3}}{\\partial W_o} } ) \\oplus \\frac{\\partial h_{t-2}}{\\partial O_{t-2}}  \\odot \\frac{\\partial O_{t-2}}{\\partial d13_{t-2}} \\odot \\frac{\\partial d13_{t-2}}{\\partial d14_{t-2}} \\odot \\frac{\\partial d14_{t-2}}{\\partial W_o} )\\oplus \\odot { \\frac{\\partial h_{t-1}}{\\partial O_{t-1}}  \\odot \\frac{\\partial O_{t-1}}{\\partial d13_{t-1}} \\odot \\frac{\\partial d13_{t-1}}{\\partial d14_{t-1}} \\odot \\frac{\\partial d14_{t-1}}{\\partial W_o} } ) \\oplus  \\odot { \\frac{\\partial h_{t}}{\\partial O_{t}}  \\odot \\frac{\\partial O_{t}}{\\partial d13_{t}} \\odot \\frac{\\partial d13_{t}}{\\partial d14_{t}} \\odot \\frac{\\partial d14_{t}}{\\partial W_o} } ) $$"
      const tequal3computehighlightedwo = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot  ( \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot \\underline{\\textcolor{blue}{\\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_{t-1}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_{t-1}}{\\partial a_{t-1}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot( \\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus { \\frac{\\partial h_{t-3}}{\\partial O_{t-3}}  \\odot \\frac{\\partial O_{t-3}}{\\partial d13_{t-3}} \\odot \\frac{\\partial d13_{t-3}}{\\partial d14_{t-3}} \\odot \\frac{\\partial d14_{t-3}}{\\partial W_o} } ) \\oplus \\frac{\\partial h_{t-2}}{\\partial O_{t-2}}  \\odot \\frac{\\partial O_{t-2}}{\\partial d13_{t-2}} \\odot \\frac{\\partial d13_{t-2}}{\\partial d14_{t-2}} \\odot \\frac{\\partial d14_{t-2}}{\\partial W_o} )\\oplus \\odot { \\frac{\\partial h_{t-1}}{\\partial O_{t-1}}  \\odot \\frac{\\partial O_{t-1}}{\\partial d13_{t-1}} \\odot \\frac{\\partial d13_{t-1}}{\\partial d14_{t-1}} \\odot \\frac{\\partial d14_{t-1}}{\\partial W_o} } ) \\oplus { \\frac{\\partial h_{t}}{\\partial O_{t}}  \\odot \\frac{\\partial O_{t}}{\\partial d13_{t}} \\odot \\frac{\\partial d13_{t}}{\\partial d14_{t}} \\odot \\frac{\\partial d14_{t}}{\\partial W_o} } ) $$"
      
      
      const dctoverawo = "$$ \\frac{\\partial C_t}{\\partial a_t} = 1 $$";
      
      const daoverdctm1wo = "$$ \\frac{\\partial a_{t}}{\\partial C_{t-1}} = \\frac{\\partial (F_t \\otimes C_{t-1})}{\\partial C_{t-1}} = diag(F_t) \\ni 1 \\times M  $$";
      
      const da1overdctm2wo = "$$ \\frac{\\partial a_{t-1}}{\\partial C_{t-2}} = \\frac{\\partial (F_{t-1} \\otimes C_{t-2})}{\\partial C_{t-2}} = diag(F_{t-1}) \\ni 1 \\times M  $$";
      
      const dctm1overdatm1 =  "$$ \\frac{\\partial C_{t-1}}{\\partial a_{t-1}} = 1 $$"
      const dLioverdH = "$$ \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t}  $$"
      const d13overdWo  = "$$  \\frac{\\partial d13}{\\partial W_o} = \\frac{\\partial d13}{\\partial d14} \\odot \\frac{\\partial d14}{\\partial W_o}  $$"
      
      return (

        <div>
            <h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <strong>FULL derivation of&nbsp;&part;E/&part;Wo</strong></h1>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;The derivation of&nbsp;<strong>&part;E/&part;Wo&nbsp;</strong>would not be any different than other derivations. It is expected to have some slight difference but overall it is the same. Let us begin with the dimension</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Dimensionality: M is defined as a hidden unit number, N is the input element number, T is the output element number</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As stated before, the summation can be interpreted by the multiplication of each row element in the weight matrix and each column element in the input vector. The overall multiplication represents the one hidden unit out of M overall hidden units.</p>

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
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Just remember that each page has its designated symbols such as d10, and d11 that are for this page only and should never be applied to other pages.</p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Next, as usual, let us write down the full derivation formula</p>
            
            <MathJaxContext config={config} version={3}>
                <MathJax inline>
                    {dEoverdWo}
                </MathJax>
            </MathJaxContext>


            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Next, let us expand on some of the terms that are not zero. In particular, we have highlighted the term that needs expansion</p>
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {dEoverdWocrossedout}
                        {dEoverdWoexpanded}
                        {dEoverdWosimplified}
                    </MathJax>
                </MathJaxContext>
                <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice above the time propogation steps are reduced to zero. There will be no back time propogation in this <strong>Wo</strong> Matrix Gradient</p>
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Here like before, let us first compute the constant term which has the timestep set to 1 or t = 1, the term will reduce to this</p>
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {dEoverdWosimplifiedcancled}
                        {dEoverdWosimplifiedcancledcont}
                    
                    </MathJax>
                </MathJaxContext>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The first two terms are commonly found in other derivation as well.Those are needed in order to compute the scaled output from hidden units. In the original LSTM paper, it only has one output so their T is assumed to be one. But here we assume the output is T.</p>
                
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

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Let us continue to derive other equations </p>
            <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {dhttoverdOttwo}
                        {dOtoverd13wo}

                    </MathJax>
                </MathJaxContext>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The next part until the next sub-title will compute the following formula</p>
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {d13overdWo}
                    </MathJax>
                </MathJaxContext>
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; And the above formula will be expanded to</p>
                <MathJaxContext config={config} version={3}>
                        <MathJax inline>
                            {dd13overd14wo}
                            {d141overdwo}
                            {d142overdWo}
                            {dd13overd14t1imesd14overdWo}
                            {dd13overd14timesd14overdWo}
                            {ddoverd1timesd21overdWocombine1}
                        </MathJax>
                </MathJaxContext>
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Since we have M hidden units, we should stack the matrix upto M rows</p>


        </div>

      )
}

export default LSTMdEoverdWo;