import './App.css';
import * as React from 'react';
import Papa from 'papaparse';
//link for the tutorial below https://www.robinwieruch.de/react-checkbox/

function App() {
  const [checked, setChecked] = React.useState(false);
  const [alarmforprice, setChecked1] = React.useState(false);
  const [pricefallalarm, setChecked2] = React.useState(false);
  const handleChange = () => {

        setChecked(!checked);
  };


  const handleChangepricealarm = () => {

    setChecked1(!alarmforprice);
};

const handlepricefallalarm = () => {

  setChecked2(!pricefallalarm);
};

const changeHandler = (event) => {
  Papa.parse(event.target.files[0], {
    header: true,
    skipEmptyLines: true,
    complete: function (results) {
      console.log(results.data)
    },
  });
};

const parseFile = file => {
  Papa.parse(file);
};

const debug_handleimportbuttonclick = () => {

  //seamingless import read csv data and display it onto the server
  //csv file reader

  Papa.parse(new File([Blob], "C:\\Users\\cheng\\Desktop\\james-study-file\\investingmentshit\\ethereumdatabase\\datafilevolclosefac1.02volratio3.0volperiod300.0timeperiod5mbelowline35.0wedgeblwline35.0rsiperiod30.0ATRperiod14.0STmultiplier2.0MA1period200.0MA2period50.0.csv"), 
  {
    complete: function(results) {
      console.log("Finished:", results.data);
    }

  });
};


  return (
    <div>
      <Checkbox
        label="ETH Or USDT"
        value={checked}
        onChange={handleChange}
      />
      <p>Is ETH? {checked.toString()}</p>

      <Checkbox
        label="Sound Alarm for bought price"
        value={alarmforprice}
        onChange={handleChangepricealarm}
      />
      <p>Sound alarm for reaching bought price? {alarmforprice.toString()}</p>
      <br></br>

      <Checkbox
        label="Sound Alarm for drop price from recent high greater than 6 percent and below bought price of course"
        value={pricefallalarm}
        onChange={handlepricefallalarm}
      />
      <p>Sound Alarm for drop price from recent high greater than 6 percent? {pricefallalarm.toString()}</p>
      
      <label>
      <Textfield
        id="BoughtInUSD"
      ></Textfield>  
      Price bought or sold 
      </label>

      <br></br>

    <div>
      {/* File Uploader */}
      <input
        type="file"
        name="file"
        accept=".csv"
        onChange={changeHandler}
        style={{ display: "block", margin: "10px auto" }}
      />
    </div>


    </div>
  );
};

const Checkbox = ({ label, value, onChange }) => {
  return (
    <label>
      <input type="checkbox" checked={value} onChange={onChange} />
      {label}
    </label>
  );
}

const Textfield = ({id}) =>  
{
  return (
    <label>
      <input type="text" id={ "TF_" + id}/>
    </label>
  );

}


// const Button = ({id}) =>  
// {
//   return (
//     <label>
//       <input type="button" id={ "BTN_" + id}/>
//     </label>
//   );

// }

export default App;
