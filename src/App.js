import './App.css';
import * as React from 'react';

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
