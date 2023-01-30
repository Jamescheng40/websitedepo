import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';    

function LSTMdEoverdWc(){
        const config = {
            "HTML-CSS":{ 
                scale: 150
            }
          };
        //dEoverdwf
        const Ftwc = '$$ {F_t = }\\sigma_s{(\\sum_{i=0}^M W_f \\otimes x_t + bias(optional))} = \\sigma_s{( W_f \\odot x_t + bias(optional))}{= \\sigma_s{(d)} \\; {,where \\; d1=\\sigma_s(\\sum_{i=0}^M W_f \\otimes x_t) \\; is \\; sigmoid \\; and \\; d1=\\sum_{i=0}^M d2}, \\; d2=W_f \\otimes x_t}$$'; 
        const itwc = '$$ {i_t = }\\sigma_s{(\\sum_{i=0}^M W_i \\otimes x_t + bias(optional))} = \\sigma_s{( W_i \\odot x_t + bias(optional))} = \\sigma_s{(d5)} \\; {where \\; d5=\\sigma_s(\\sum_{i=0}^M W_i \\otimes x_t) \\; is \\; sigmoid \\; and \\; d6=\\sum_{i=0}^M d7}, \\; d7=W_i \\otimes x_t   $$';
        const Otwc = '$$ {O_t = }\\sigma_s{(\\sum_{i=0}^M W_o \\otimes x_t + bias(optional))} = \\sigma_s{( W_o \\odot x_t + bias(optional))} $$';
        const Cdtwc = "$$ {C'_t = }\\sigma_t{(\\sum_{i=0}^M W_c \\otimes x_t + bias(optional))} = \\sigma_s{( W_c \\odot x_t + bias(optional))}{= \\sigma_s{(d10)} \\; {,where \\; d10=\\sigma_s(\\sum_{i=0}^M W_c \\otimes x_t) \\; is \\; sigmoid \\; and \\; d11=\\sum_{i=0}^M d12}, \\; d12=W_c \\otimes x_t \\; where \\; \\sigma_t \\; is \\; tanh} $$";
        const Ctwc =  "$$ {C_t = }  F_t \\otimes {(C_{t-1})} \\oplus i_t \\otimes C'_t {, \\; where \\; \\otimes \\; is \\; element-wise \\; multiply \\; and \\oplus \\; is \\; element-wise \\; addition  ,where \\; a=F_t \\otimes {(C_{t-1})} \\; and \\; a1= i_t \\otimes C'_t }$$";
        const htwc  = "$$ {h_t = }  O_t \\otimes \\sigma_t{(C_t)} = {O_t \\otimes b_t} {, \\; where \\; b_t \\; is \\; \\sigma_t{(C_t)}} $$";
        const Liwc = "$$ {L_t =  W_v \\odot h_t =}{\\sum_{i=0}^T W_v \\otimes h_t + bias(optional)}, \\; where \\; L1 =\\sum_{i=0}^M L2,and \\; L2=W_v \\otimes h_t $$"
        const Piwc = "$$ {P_i = softmax(L_i)}{=\\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}} } $$"
        const Eiwfc = "$$ {E_i=}{- \\sum_{i=0}^T Y_i  \\otimes log(P_i) + bias(optional)}{= - \\sum_{i=0}^T g,where \\; g = Y_i  \\otimes log(P_i)} $$"
        const Ftgeneral = '$$ {F_t = }\\sigma_s{(\\sum_{i=0}^M W_f \\otimes x_t + bias(optional))} = \\sigma_s{( W_f \\odot x_t + bias(optional))}{= \\sigma_s{(d1)} \\; {,where \\; d1=\\sum_{i=0}^M W_f \\otimes x_t \\; is \\; sigmoid \\; and \\; d1=\\sum_{i=0}^M d2}, \\; d2=W_f \\otimes x_t}$$'; 
        const itgeneral = '$$ {i_t = }\\sigma_s{(\\sum_{i=0}^M W_i \\otimes x_t + bias(optional))} = \\sigma_s{( W_i \\odot x_t + bias(optional))} = \\sigma_s{(d6)} \\; {where \\; d6=\\sum_{i=0}^M W_i \\otimes x_t \\; is \\; sigmoid \\; and \\; d6=\\sum_{i=0}^M d7}, \\; d7=W_i \\otimes x_t   $$';
        const Otgeneral = '$$ {O_t = }\\sigma_s{(\\sum_{i=0}^M W_o \\otimes x_t + bias(optional))} = \\sigma_s{( W_o \\odot x_t + bias(optional)) = \\sigma_s{(d13)}} {where \\; d13=\\sum_{i=0}^M W_O \\otimes x_t \\; is \\; sigmoid \\; and \\; d13=\\sum_{i=0}^M d14}, \\; d14=W_o \\otimes x_t  $$';
        const Cdtgeneral = "$$ {C'_t = }\\sigma_t{(\\sum_{i=0}^M W_c \\otimes x_t + bias(optional))} = \\sigma_t{( W_c \\odot x_t + bias(optional))}{= \\sigma_t{(d11)} \\; {,where \\; d11=\\sum_{i=0}^M W_c \\otimes x_t \\; is \\; sigmoid \\; and \\; d11=\\sum_{i=0}^M d12}, \\; d12=W_c \\otimes x_t \\; where \\; \\sigma_t \\; is \\; tanh} $$";
        const Ctgeneral =  "$$ {C_t = }  F_t \\otimes {(C_{t-1})} \\oplus i_t \\otimes C'_t {, \\; where \\; \\otimes \\; is \\; element-wise \\; multiply \\; and \\oplus \\; is \\; element-wise \\; addition  ,where \\; a=F_t \\otimes {(C_{t-1})} \\; and \\; a1= i_t \\otimes C'_t }$$";
        const htgeneral  = "$$ {h_t = }  O_t \\otimes \\sigma_t{(C_t)} = {O_t \\otimes b_t} {, \\; where \\; b_t \\; is \\; \\sigma_t{(C_t)}} $$";
        const Ligeneral = "$$ {L_t =  W_v \\odot h_t =}{\\sum_{i=0}^T W_v \\otimes h_t + bias(optional)}, \\; where \\; L1 =\\sum_{i=0}^M L2,and \\; L2=W_v \\otimes h_t $$"
        const Pigeneral = "$$ {P_i = softmax(L_i)}{=\\frac{e^{L_i}}{\\sum_{i=0}^T e^{L_k}} } $$"
        const Eigeneral = "$$ {E_i=}{- \\sum_{i=0}^T Y_i  \\otimes log(P_i) + bias(optional)}{= - \\sum_{i=0}^T g,where \\; g = Y_i  \\otimes log(P_i)} $$"
        
        
        const dEoverdWc = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_c} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_c} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_c} \\oplus \\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_c} ) \\oplus \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_c}))}    $$";
        const dEoverdWccrossedout = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_c} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_c}} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_c} \\oplus \\cancelto{0}{\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_c}} ) \\oplus \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_c}))}    $$";
        const dEoverdG_j_final = "$$ \\frac{\\partial E}{\\partial g_j} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times T   $$"

        const dgoverdP_i_final = "$$ \\frac{\\partial g_j}{\\partial P_i} = diag(Y_i/P_i) \\ni T \\times T  $$"
    
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

        
        const dhtoverdb = "$$\\frac{\\partial h_t}{\\partial b_t} = \\frac{\\partial \\begin{bmatrix} O_1 \\times b_1 \\\\ O_2 \\times b_2 \\\\ \\vdots \\\\ O_M \\times b_M \\end{bmatrix}}{\\partial b_i} = \\begin{bmatrix} \\frac{\\partial (O_1 \\times b_1)}{\\partial b_1} & \\cancelto{0}{\\frac{\\partial (O_1 \\times b_1)}{\\partial b_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (O_1 \\times b_1)}{\\partial b_M}} \\\\ \\cancelto{0}{\\frac{\\partial (O_2 \\times b_2)}{\\partial b_1}} & \\frac{\\partial (O_2 \\times b_2)}{\\partial b_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (O_2 \\times b_2)}{\\partial b_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (O_M \\times b_M)}{\\partial b_1}} & \\cancelto{0}{\\frac{\\partial (O_M \\times b_M)}{\\partial b_2}} & \\cdots & \\frac{\\partial (O_M \\times b_M)}{\\partial b_M}  \\end{bmatrix} = diag(O_t) \\ni M \\times M $$"
      
        const dboverdCt = "$$\\frac{\\partial b_t}{\\partial C_t} = \\frac{\\partial \\begin{bmatrix} \\tanh(C_1) \\\\ \\tanh(C_2) \\\\ \\vdots \\\\ \\tanh(C_M) \\end{bmatrix}}{\\partial C_t} = \\begin{bmatrix} \\frac{\\partial (\\tanh(C_1))}{\\partial C_1} & \\cancelto{0}{\\frac{\\partial (\\tanh(C_1))}{\\partial C_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (\\tanh(C_1))}{\\partial C_M}} \\\\ \\cancelto{0}{\\frac{\\partial (\\tanh(C_2))}{\\partial C_1}} & \\frac{\\partial (\\tanh(C_2))}{\\partial C_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (\\tanh(C_2))}{\\partial C_M}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (\\tanh(C_M))}{\\partial C_1}} & \\cancelto{0}{\\frac{\\partial (\\tanh(C_M))}{\\partial C_2}} & \\cdots & \\frac{\\partial (tanh(C_M))}{\\partial C_M}  \\end{bmatrix} = 1 - \\tanh^2(C_t) \\ni M \\times M $$"        
        
        const dEoverdWcexpanded = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_c} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot ( \\cancelto{0}{\\frac{\\partial h_t}{\\partial O_t} \\odot \\frac{\\partial O_t}{\\partial W_c}} \\oplus \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot (\\frac{\\partial C_t}{\\partial a_t} \\odot (\\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_c} \\oplus \\cancelto{0}{\\frac{\\partial a_t}{\\partial F_t} \\odot \\frac{\\partial F_t}{\\partial W_c}} ) \\oplus \\underline{\\textcolor{blue}{\\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial W_c}))}}}   $$";
        const dEoverdWcsimplified = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_c} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot   \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} ( \\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot \\frac{\\partial C_\{t-1\}}{\\partial W_c} \\oplus { \\frac{\\partial C_t}{\\partial a1_t}  \\odot \\frac{\\partial a1_t}{\\partial C'_t} \\odot \\frac{\\partial C'_t}{\\partial d11} \\odot \\frac{\\partial d11}{\\partial d12} \\odot \\frac{\\partial d12}{\\partial W_c}}) }    $$";
        const dEoverdWcsimplifiedcancled = "$$ \\frac{\\partial E(P_i,L_i,h_t,C_t,i_t,C_\{t-1\})}{\\partial W_c} {=} \\frac{\\partial E}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial C_t} \\odot \\frac{\\partial C_t}{\\partial F_{c_t}}{= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot   \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} (  \\cancelto{0}{\\odot \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_{t-1}} \\odot \\frac{\\partial C_{t-1}}{\\partial W_c}} \\oplus { \\frac{\\partial C_t}{\\partial a1_t}  \\odot \\frac{\\partial a1_t}{\\partial C'_t} \\odot \\frac{\\partial C'_t}{\\partial d11} \\odot \\frac{\\partial d11}{\\partial d12} \\odot \\frac{\\partial d12}{\\partial W_c}}) }    $$";

        const dEoverdWcsimplifiedcancledcont = "$$ {= \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot   \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot  { \\frac{\\partial C_t}{\\partial a1_t}  \\odot \\frac{\\partial a1_t}{\\partial C'_t} \\odot \\frac{\\partial C'_t}{\\partial d11} \\odot \\frac{\\partial d11}{\\partial d12} \\odot \\frac{\\partial d12}{\\partial W_c}} }    $$";
       
        const dCtoverda1twc = "$$ \\frac{\\partial C_t}{\\partial a1_t} = 1 $$"
        const da1toverdCdtwc = "$$ \\frac{\\partial a1_t}{\\partial C'_t} =  \\frac{\\partial \\begin{bmatrix} i_{{t}_1} \\times C'_{{t}_1} \\\\ i_{{t}_2} \\times C'_{{t}_2} \\\\ \\vdots \\\\ i_{{t}_M} \\times C'_{{t}_M} \\end{bmatrix}}{\\partial C'_t} = diag(i_{t}) \\ni M \\times M $$"
        const dcdtoverd10 = "$$ \\frac{\\partial C'_t}{\\partial d11_t} = \\frac{\\partial \\begin{bmatrix} \\tanh(d11_{{t}_1}) \\\\ \\tanh(d11_{{t}_2}) \\\\ \\vdots \\\\ \\tanh(d11_{{t}_M}) \\end{bmatrix}}{\\partial d11_t} = 1 - \\tanh^2(d11_t) \\ni M \\times M $$"

        const dd11overd12 = "$$ \\frac{\\partial d11_1}{\\partial d12_1} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N; \\; \\frac{\\partial d11_2}{\\partial d12_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N \\cdots \\frac{\\partial d11_N}{\\partial d12_N} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times N $$"

        const d12overdwc = "$$ \\frac{\\partial d12}{\\partial W_c} = \\frac{\\partial \\begin{bmatrix} W_{{c}_{11}} \\times x_1 \\\\ W_{{c}_{12}} \\times x_2 \\\\ \\vdots \\\\ W_{{c}_{1N}} \\times x_N \\end{bmatrix}}{\\partial W_c}  = \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix}  =  \\ni M \\times M $$ "
        const d22overdWc = "$$ \\frac{\\partial d22}{\\partial W_c} = \\frac{\\partial \\begin{bmatrix} W_{{c}_{21}} \\times x_1 \\\\ W_{{c}_{22}} \\times x_2 \\\\ \\vdots \\\\ W_{{c}_{2N}} \\times x_N \\end{bmatrix}}{\\partial W_c}  = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix}  =  \\ni N \\times MN $$ "
        
        const dd10overd1t1imesd11overdWc = "$$ \\frac{\\partial d11_1}{\\partial d12_1} \\odot \\frac{\\partial d12_1}{\\partial W_c} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} x_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & x_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & x_N & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
        const ddoverd12timesd21overdWc = "$$ \\frac{\\partial d11_2}{\\partial d12_2} \\odot \\frac{\\partial d12_2}{\\partial W_c} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & x_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & x_N & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MN $$"
        const ddoverd1timesd21overdWccombine1 = "$$ \\frac{\\partial d11}{\\partial W_c} = \\frac{\\partial d11}{\\partial d12} \\odot \\frac{\\partial d12}{\\partial W_c} =  \\begin{bmatrix} x_1 & x_2 & \\cdots & x_N & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & x_1 & x_2 & \\cdots & x_N & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & x_1 & x_2 & \\cdots & x_N \\end{bmatrix} \\ni M \\times MN  $$"
        
        const tequal3computenohighlightedwc = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}} \\odot ( \\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}} \\odot ( {\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}} \\odot(\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}} \\oplus \\frac{\\partial C_{t-3}}{\\partial a1_{t-3}} \\odot \\frac{\\partial a1_{t-3}}{\\partial C'_{t-3}} \\odot \\frac{\\partial C'_{t-3}}{\\partial d11_{t-3}} \\odot \\frac{\\partial d11_{t-3}}{\\partial d12_{t-3}} \\odot \\frac{\\partial d12_{t-3}}{\\partial W_{C_{t-3}}}) \\oplus \\frac{\\partial C_{t-2}}{\\partial a1_{t-2}} \\odot \\frac{\\partial a1_{t-2}}{\\partial C'_{t-2}} \\odot \\frac{\\partial C'_{t-2}}{\\partial d11_{t-2}} \\odot \\frac{\\partial d11_{t-2}}{\\partial d12_{t-2}} \\odot \\frac{\\partial d12_{t-2}}{\\partial W_{C_{t-2}}} )\\oplus \\frac{\\partial C_{t-1}}{\\partial a1_{t-1}} \\odot \\frac{\\partial a1_{t-1}}{\\partial C'_{t-1}} \\odot \\frac{\\partial C'_{t-1}}{\\partial d11_{t-1}} \\odot \\frac{\\partial d11_{t-1}}{\\partial d12_{t-1}} \\odot \\frac{\\partial d12_{t-1}}{\\partial W_{C_{t-1}}} ) \\oplus  \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial C'_t} \\odot \\frac{\\partial C'_t}{\\partial d11} \\odot \\frac{\\partial d11}{\\partial d12} \\odot \\frac{\\partial d12}{\\partial W_c} ) $$"
        const tequal3computehighlightedwc = " $$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial P_i} \\odot \\frac{\\partial P_i}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t} \\odot \\frac{\\partial h_t}{\\partial b_t} \\odot \\frac{\\partial b_t}{\\partial C_t} \\odot ( \\underline{ \\textcolor{blue}{\\frac{\\partial C_t}{\\partial a_t} \\odot \\frac{\\partial a_t}{\\partial C_\{t-1\}}}} \\odot ( \\underline{\\textcolor{blue}{\\frac{\\partial C_\{t-1\}}{\\partial a_\{t-1\}} \\odot \\frac{\\partial a_{t-1}}{\\partial C_{t-2}}}} \\odot ( \\underline{\\textcolor{blue}{{\\frac{\\partial a_{t-2}}{\\partial C_{t-2}} \\odot \\frac{\\partial a_{t-2}}{\\partial C_{t-3}}}}} \\odot(\\cancelto{0}{\\frac{\\partial C_{t-3}}{\\partial a_{t-3}} \\odot \\frac{\\partial a_{t-3}}{\\partial C_{t-4}}} \\oplus \\frac{\\partial C_{t-3}}{\\partial a1_{t-3}} \\odot \\frac{\\partial a1_{t-3}}{\\partial C'_{t-3}} \\odot \\frac{\\partial C'_{t-3}}{\\partial d11_{t-3}} \\odot \\frac{\\partial d11_{t-3}}{\\partial d12_{t-3}} \\odot \\frac{\\partial d12_{t-3}}{\\partial W_{C_{t-3}}}) \\oplus \\frac{\\partial C_{t-2}}{\\partial a1_{t-2}} \\odot \\frac{\\partial a1_{t-2}}{\\partial C'_{t-2}} \\odot \\frac{\\partial C'_{t-2}}{\\partial d11_{t-2}} \\odot \\frac{\\partial d11_{t-2}}{\\partial d12_{t-2}} \\odot \\frac{\\partial d12_{t-2}}{\\partial W_{C_{t-2}}} )\\oplus \\frac{\\partial C_{t-1}}{\\partial a1_{t-1}} \\odot \\frac{\\partial a1_{t-1}}{\\partial C'_{t-1}} \\odot \\frac{\\partial C'_{t-1}}{\\partial d11_{t-1}} \\odot \\frac{\\partial d11_{t-1}}{\\partial d12_{t-1}} \\odot \\frac{\\partial d12_{t-1}}{\\partial W_{C_{t-1}}} ) \\oplus  \\frac{\\partial C_t}{\\partial a1} \\odot \\frac{\\partial a1}{\\partial C'_t} \\odot \\frac{\\partial C'_t}{\\partial d11} \\odot \\frac{\\partial d11}{\\partial d12} \\odot \\frac{\\partial d12}{\\partial W_c} ) $$"

        const dctm1overattm1 = "$$ \\frac{\\partial C_{t-1}}{\\partial a1_{t-1}} = 1 $$"
       
        const da1tm1overdcdtm1 = "$$ \\frac{\\partial a_{t-1}}{\\partial C'_{t-1}} = \\frac{\\partial \\begin{bmatrix} F_{{t}_1} \\times C'_{{t-1}_1} \\\\ F_{{t}_2} \\times C'_{{t-1}_2} \\\\ \\vdots \\\\ F_{{t}_M} \\times C'_{{t-1}_M} \\end{bmatrix}}{\\partial C'_{t-1}} = diag(F_t)  $$"
        const dLioverdH = "$$ \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial h_t}  $$"
        const d11overdWc  = "$$ \\frac{\\partial d11}{\\partial W_c} = \\frac{\\partial d11}{\\partial d12} \\odot \\frac{\\partial d12}{\\partial W_c}  $$"
        

        return (
        <div>
<h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <strong>Full derivation of&nbsp;&part;E/&part;Wc</strong></h1>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;The derivation of&nbsp;&nbsp;<strong>&part;E/&part;Wc&nbsp;</strong>would have a slightly different equation than the rest of the post, but overall the concept is the same and they have a lot of overlapping equations. Let us see below for the equations</p>

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
     {Pigeneral}
     {Eigeneral}
    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Just remember that each page has its designated symbols such as d10, and d11 that are for this page only and should never be applied to other pages.</p>

<p>&nbsp; &nbsp;</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Next, as usual, let us write down the full derivation formula</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
        {dEoverdWc}
    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Next, let us expand on some of the terms that are not zero. In particular, we have highlighted the term that needs expansion</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
        {dEoverdWccrossedout}
        {dEoverdWcexpanded}
    </MathJax>
</MathJaxContext>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Here like before, let us first compute the constant term which has the timestep set to 1 or t = 1, the term will reduce to this</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
        {dEoverdWcsimplified}
        {dEoverdWcsimplifiedcancled}
        {dEoverdWcsimplifiedcancledcont}
    </MathJax>
</MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The first part in blue has already been computed please check out this <a href="http://post">post</a> or this <a href="http://post1">post</a>&nbsp;for the full derivation of those equations. For this post, we will keep it simple and use only the final product from the derivation. The derivation is based on the softmax and logit. Those are needed in order to compute the scaled output from hidden units. In the original LSTM paper, it only has one output so their T is assumed to be one. But here we assume the output is T.</p>

        <MathJaxContext config={config} version={3}>
            <MathJax inline>
                {dEoverdG_j_final}
                {dgoverdP_i_final}
                {dPoverdLi_final}
                {kronica}

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
       {dhtoverdb}
       {dboverdCt}
       {dCtoverda1twc}
       {da1toverdCdtwc}
       {dcdtoverd10}
  
    </MathJax>
</MathJaxContext>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The next part until the next sub-title will compute the following formula</p>
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {d11overdWc}
                    </MathJax>
                </MathJaxContext>
                <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; And the above formula will be expanded to</p>
             
                {dd11overd12}
                <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                        {d12overdwc}
                        {d22overdWc}
                        {dd10overd1t1imesd11overdWc}
                        {ddoverd12timesd21overdWc}
                        {ddoverd1timesd21overdWccombine1}
                    </MathJax>
                </MathJaxContext>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Notice that we should refer to the d10 variable in this page only. Do not use other page for d10 reference</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;</p>

<p>&nbsp;</p>

<p>&nbsp;</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Since we have M hidden units, we should stack the matrix upto M rows</p>
<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Now let us try to see if we are at a different timestep and how the equations are going to expand out. Some of the equations would go deeper than initially would be. Let us see how this expanded out.</p>


<MathJaxContext config={config} version={3}>
    <MathJax inline>
        {tequal3computenohighlightedwc}
        {tequal3computehighlightedwc}
    </MathJax>
</MathJaxContext>


<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Notice that there are extra 2 partial differential equations at the beginning of the blue color highlight. This is the only exception for Wf, and the rest for t=4, t=5, t=6 will have the same rate of adding two extra equations per line.</p>

<MathJaxContext config={config} version={3}>
    <MathJax inline>
        {dctm1overattm1}
        {da1tm1overdcdtm1}
    </MathJax>
</MathJaxContext>

        </div>

                )

}


export default LSTMdEoverdWc;
