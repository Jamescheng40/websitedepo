import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';

function LSTMdEoverdWv(){
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
    
    const dEoverdG_j_final = "$$ \\frac{\\partial E}{\\partial g_j} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times T   $$"
            
    const dgoverdL_i_final = "$$ \\frac{\\partial g_j}{\\partial L_t} = \\frac{\\partial \\begin{bmatrix} (L_1 - Y_1)^2 \\\\ (L_2 - Y_2)^2 \\\\ \\vdots \\\\ (L_T - Y_T)^2 \\end{bmatrix}}{\\partial L_t} = \\begin{bmatrix} \\frac{\\partial (L_1 - Y_1)^2}{\\partial L_1} & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2}} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_1 - Y_1)^2}{\\partial L_T}} \\\\ \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_1}} & \\frac{\\partial (L_2 - Y_2)^2}{\\partial L_2} & \\cdots & \\cancelto{0}{\\frac{\\partial (L_2 - Y_2)^2}{\\partial L_T}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}} & \\cancelto{0}{\\frac{\\partial (L_T - Y_T)^2}{\\partial L_2}} & \\cdots & \\frac{\\partial (L_T - Y_T)^2}{\\partial L_T}  \\end{bmatrix} = diag(2 \\times (L_1 - Y_1)) \\ni T \\times T  $$"
    
    const dgoverdL_i_finalcont1 = "$$ \\frac{\\partial (L_1 - Y_1)^2}{\\partial L_1} = 2 \\times (L_1 - Y_1) \\times (\\cancelto{1}{\\frac{\\partial L_1}{\\partial L_1}} - \\cancelto{0}{\\frac{\\partial Y_1}{\\partial L_1}})   $$"

    //const dPoverdLi_final = "$$\\frac{\\partial P_i}{\\partial L_i} = \\begin{bmatrix} P_1 \\times (1 - P_1) & - P_1 \\times P_2 & \\cdots & -P_1 \\times P_T \\\\ -P_2 \\times P_1 & P_2 \\times (1 - P_2) & \\cdots & P_2 \\times P_T  \\\\ \\vdots & \\vdots & \\vdots & \\vdots \\\\ -P_T \\times P_1 & -P_T \\times P_2 & \\cdots & P_T \\times (1 - P_T) \\end{bmatrix} = P_i \\times (\\delta_{ij} - P_j ) $$"
    //const kronica = "$$ \\delta_{ij} = \\begin{cases}1, &         \\text{if } i=j,\\\\0, &  \\text{if } i\\neq j.\\end{cases} $$"
    
        
     const dEoverdwv = "$$  \\frac{\\partial E}{\\partial g} \\odot \\frac{\\partial g}{\\partial L_i} \\odot \\frac{\\partial L_i}{\\partial L1} \\odot \\frac{\\partial L1_i}{\\partial L2_i} \\odot \\frac{\\partial L2_i}{\\partial W_v} $$"

     const ddoverd1Wv = "$$  \\frac{\\partial L1_1}{\\partial L2_1} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M; \\; \\frac{\\partial L1_2}{\\partial L2_2} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M \\cdots \\frac{\\partial L1_T}{\\partial L2_T} =  \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\ni 1 \\times M $$"
    
     const d21overdWv = "$$ \\frac{\\partial L2_1}{\\partial W_v} = \\frac{\\partial \\begin{bmatrix} W_{{v}_{11}} \\times h_1 \\\\ W_{{v}_{12}} \\times h_2 \\\\ \\vdots \\\\ W_{{v}_{1M}} \\times h_M \\end{bmatrix}}{\\partial W_v} = \\begin{bmatrix} \\frac{\\partial ( W_{{v}_{11}} \\times h_1 )}{\\partial W_{{v}_{11}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{1M}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{11}} \\times h_1)}{\\partial W_{{v}_{MT}}}} & \\cdots & 0 \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{11}}}} & \\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{12}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{1M}}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{MT}}}} & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{12}}}} & \\cdots & \\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{1M}}} & 0 & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{1M}} \\times h_M)}{\\partial W_{{v}_{MT}}}} & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} h_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & h_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & h_M & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix}  \\ni M \\times MT $$ "
     const d22overdWv = "$$ \\frac{\\partial L2_2}{\\partial W_v} = \\frac{\\partial \\begin{bmatrix} W_{{v}_{11}} \\times h_1 \\\\ W_{{v}_{12}} \\times h_2 \\\\ \\vdots \\\\ W_{{v}_{1M}} \\times h_M \\end{bmatrix}}{\\partial W_v} = \\begin{bmatrix} \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{1M}}}} & \\frac{\\partial ( W_{{v}_{21}} \\times h_1 )}{\\partial W_{{v}_{21}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{22}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{21}} \\times h_1)}{\\partial W_{{v}_{MT}}}}  \\\\ \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{1M}}}} & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{21}}}} & \\frac{\\partial ((W_{{v}_{22}} \\times h_2))}{\\partial W_{{v}_{22}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{2N}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial ((W_{{v}_{12}} \\times h_2))}{\\partial W_{{v}_{MT}}}} \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{11}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{12}}}} & \\cdots & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{1M}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{21}}}} & \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{22}}}} & 0 & \\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{2N}}} & \\cdots &  \\cancelto{0}{\\frac{\\partial (W_{{v}_{2N}} \\times h_M)}{\\partial W_{{v}_{MT}}}}  \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & h_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & h_M & \\cdots & 0 \\end{bmatrix}  \\ni M \\times MT $$ "
     
     const ddoverd1timesd21overdWv = "$$ \\frac{\\partial L1_1}{\\partial L2_1} \\odot \\frac{\\partial L2_1}{\\partial W_v} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} h_1 & 0 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & h_2 & \\cdots & 0 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & h_M & 0 & \\cdots & 0 & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MT $$"
     const ddoverd12timesd21overdWv = "$$ \\frac{\\partial L1_2}{\\partial L2_2} \\odot \\frac{\\partial L2_2}{\\partial W_v} = \\begin{bmatrix} 1 & 1 & \\cdots & 1 \\end{bmatrix} \\odot \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & 0 & \\cdots & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & 0 & h_2 & \\cdots &  0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & h_M & \\cdots & 0 \\end{bmatrix} = \\begin{bmatrix} 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots & 0   \\end{bmatrix} \\ni 1 \\times MT $$"
     const ddoverd1timesd21overdWvcombine = "$$ \\begin{bmatrix} h_1 & h_2 & \\cdots & h_M & 0 & 0 & \\cdots & 0 & \\cdots & 0 & 0 & \\cdots & 0 \\\\ 0 & 0 & \\cdots & 0 & h_1 & h_2 & \\cdots & h_M & \\cdots &  0 & 0 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots & \\vdots& \\vdots & \\vdots& \\vdots & \\vdots\\\\ 0 & 0 & \\cdots & 0 & 0 & 0 & \\cdots & 0 & \\cdots & h_1 & h_2 & \\cdots & h_M \\end{bmatrix} \\ni T \\times MT  $$"
     
    return (
        <div>
            <h1>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<strong>FULL&nbsp;</strong><span style={{fontSize:37.379999999999995}}><strong>Derivation of&nbsp;</strong></span><strong>&nbsp;&part;E/&part;W<span style={{fontSize:21.36}}>v</span></strong></h1>

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
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;As you can see from above, the&nbsp;<strong>Wv&nbsp;</strong>is only several equations away from the <strong>E</strong>. Specifically, we can use one notation to summarize the dependency,&nbsp;<strong>&part;E(Pi, Lᵢ,)/&part;Wv</strong></p>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;So the number of equations here will not be as overwhelming as the other post where we have millions of equations since there are only two dependencies here</p>
            <MathJaxContext config={config} version={3}>
                    <MathJax inline>
            {dEoverdwv}
            </MathJax>
                </MathJaxContext>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Since there are only two dependencies, the equations expansion will be straightforward and very easy. Let us take a look at how it expands for the first three terms. Detailed explanation please check out&nbsp;<a href="http://a">this</a> and <a href="http://b">this</a>. They are already computed in those posts so it would be straightforward to understand what it is going on.&nbsp;</p>
            <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                    {dEoverdG_j_final}
                    {dgoverdL_i_final}
                    {dgoverdL_i_finalcont1}
                    </MathJax>
                </MathJaxContext>
            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Finally, let us compute the last three terms.&nbsp;</p>

            <MathJaxContext config={config} version={3}>
                    <MathJax inline>
                    
                    {ddoverd1Wv}
                    {d21overdWv}
                    {d22overdWv}
                    {ddoverd1timesd21overdWv}
                    {ddoverd12timesd21overdWv}
                    {ddoverd1timesd21overdWvcombine}
                    </MathJax>
                </MathJaxContext>

            <p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;There we have it, the full derivation of&nbsp;<strong>&part;E/&part;W<span style={{fontSize:10.68}}>v.&nbsp;</span></strong>Be sure to check out other derivations from here.</p>

        </div>
    )

}


export default LSTMdEoverdWv;