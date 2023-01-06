import React from "react";
import PropTypes from "prop-types";

import { scaleTime } from "d3-scale";
import { utcDay } from "d3-time";

import { ChartCanvas, Chart } from "react-stockcharts";
import { CandlestickSeries } from "react-stockcharts/lib/series";
import { XAxis, YAxis } from "react-stockcharts/lib/axes";
import { fitWidth } from "react-stockcharts/lib/helper";
import { last, timeIntervalBarWidth } from "react-stockcharts/lib/utils";

import {BarSeries, RSISeries, LineSeries} from "react-stockcharts/lib/series";
import {
	CrossHairCursor,
	MouseCoordinateY} from "react-stockcharts/lib/coordinates";
import { format } from "d3-format";

import {
	SingleValueTooltip,
} from "react-stockcharts/lib/tooltip";

class CandleStickChart extends React.Component {
	render() {
		const { type, width, data, ratio } = this.props;
		const xAccessor = d => d.date;
		const xExtents = [
			xAccessor(last(data)),
			xAccessor(data[data.length - 100])
		];
		return (
			<div>
			<ChartCanvas 
					height={1700}
					ratio={ratio}
					width={width}
					margin={{ left: 50, right: 50, top: 10, bottom: 30 }}
					type={type}
					seriesName="MSFT"
					data={data}
					xAccessor={xAccessor}
					xScale={scaleTime()}
					xExtents={xExtents}>

				<Chart id={1} height={400} yExtents={d => [d.high, d.low]} padding={{ top: 10, bottom: 20 }}  >
					<XAxis axisAt="bottom" orient="bottom" ticks={6}/>
					<YAxis axisAt="left" orient="left" ticks={5} />

					<MouseCoordinateY
						at="right"
						orient="right"
						displayFormat={format(".2f")} />

					<LineSeries yAccessor={d => d.pricema}/>
					{/* <LineSeries yAccessor={d => d.pricema2}/> */}
					<LineSeries yAccessor={d => d.EMA2S } stroke={"#ff00ff"}/>


					{/* <LineSeries yAccessor={d => d.CElongS}/>
					<LineSeries yAccessor={d => d.CEShortS}/> */}
					<LineSeries yAccessor={d => d.STfinalupS}/>
					<LineSeries yAccessor={d => d.STfinaldownS}/>

					{/* Todo:hard coded width make it depend on the time */}
					<CandlestickSeries width={2} candletype={2}/>
				</Chart>

				<Chart id={2} height={400}  origin={(w, h) => [0, h - 530]}  yExtents={d => [d.high, d.low]} padding={{ top: 10, bottom: 20 }} >
					<XAxis axisAt="bottom" orient="bottom" ticks={6}/>
					<YAxis axisAt="left" orient="left" ticks={5} />

					<LineSeries yAccessor={d => d.STfinalupS}/>
					<LineSeries yAccessor={d => d.STfinaldownS}/>

					<LineSeries yAccessor={d => d.pricema}/>
					{/* <LineSeries yAccessor={d => d.pricema2}/> */}
					<LineSeries yAccessor={d => d.EMA2S} stroke={"#ff00ff"}/>

					<MouseCoordinateY
						at="right"
						orient="right"
						displayFormat={format(".2f")} />


					{/* Todo:hard coded width make it depend on the time */}
					<CandlestickSeries width={2} candletype={1}/>
				</Chart>

				<div></div>

				{/* subplot example and RSI value from the csv file for different indicator for more example configuration checkout https://codesandbox.io/s/github/rrag/react-stockcharts-examples2/tree/master/examples/CandleStickChartWithRSIIndicator?file=/src/Chart.js */}
				<Chart id={3}
					yExtents={[0,100]}
					height={125} origin={(w, h) => [0, h - 800]}
				>
					<XAxis axisAt="bottom" orient="bottom" showTicks={false} outerTickSize={0} />
					<YAxis axisAt="right"
						orient="right"
						tickValues={[30, 50, 70]}/>
					<MouseCoordinateY
						at="right"
						orient="right"
						displayFormat={format(".2f")} />

					<RSISeries yAccessor={d => d.RSIratio} />

				</Chart>
				
				<Chart id={4}
					yExtents={d => d.volume}
					height={125} origin={(w, h) => [0, h - 125]}
				>
					<XAxis axisAt="bottom" orient="bottom" showTicks={false} outerTickSize={0} />
					<YAxis axisAt="right"
						orient="right"
						tickValues={[30, 50, 70]}/>
					<MouseCoordinateY
						at="right"
						orient="right"
						displayFormat={format(".2f")} />

					<LineSeries yAccessor={d => d.volume} />

				</Chart>

				<Chart id={5}
					yExtents={[0,400]}
					height={125} origin={(w, h) => [0, h - 1300]}
				>
					<XAxis axisAt="bottom" orient="bottom" showTicks={false} outerTickSize={0} />
					<YAxis axisAt="right"
						orient="right"
						tickValues={[25]}/>
					<MouseCoordinateY
						at="right"
						orient="right"
						displayFormat={format(".2f")} />

					<RSISeries yAccessor={d => d.pricevolS} />
					<SingleValueTooltip
						yAccessor={d => d.pricevolS}
						yLabel={`pricevolatility`}
						yDisplayFormat={format(".2f")}
						/* valueStroke={atr14.stroke()} - optional prop */
						/* labelStroke="#4682B4" - optional prop */
						origin={[-40, 15]}/>
				</Chart>

				<Chart id={11}
					yExtents={d => d.ATRwriteS}
					height={125} origin={(w, h) => [0, h - 1150]}
				>
					<XAxis axisAt="bottom" orient="bottom" showTicks={false} outerTickSize={0} />
					<YAxis axisAt="right"
						orient="right"
						tickValues={[25]}/>
					<MouseCoordinateY
						at="right"
						orient="right"
						displayFormat={format(".2f")} />

					<RSISeries yAccessor={d => d.ATRwriteS} />
					<SingleValueTooltip
						yAccessor={d => d.ATRwriteS}
						yLabel={`ATR Value`}
						yDisplayFormat={format(".2f")}
						/* valueStroke={atr14.stroke()} - optional prop */
						/* labelStroke="#4682B4" - optional prop */
						origin={[-40, 15]}/>
				</Chart>

				<Chart id={6}
					yExtents={d => d.volmavariance}
					height={125} origin={(w, h) => [0, h - 1000]}
				>
					<XAxis axisAt="bottom" orient="bottom" showTicks={false} outerTickSize={0} />
					<YAxis axisAt="right"
						orient="right"
						tickValues={[30, 50, 70]}/>
					<MouseCoordinateY
						at="right"
						orient="right"
						displayFormat={format(".2f")} />

					<LineSeries yAccessor={d => d.volmavariance} />
					<SingleValueTooltip
						yAccessor={d => d.volmavariance}
						yLabel={`volvola`}
						yDisplayFormat={format(".2f")}
						/* valueStroke={atr14.stroke()} - optional prop */
						/* labelStroke="#4682B4" - optional prop */
						origin={[-40, 15]}/>
				</Chart>

				<Chart id={7} height={400}
					yExtents={[d => d.sighdvertrsiS]}
					origin={(w, h) => [0, h - 530]}
				>
					{/* Blue */}
					<BarSeries yAccessor={d => d.blwlineplace} fill={"#3346FF"}/>
				</Chart>

				<Chart id={8} height={400}
					yExtents={[d => d.sighdvertrsiS]}
					origin={(w, h) => [0, h - 530]}
				>
					{/* purple */}
					<BarSeries yAccessor={d => d.botwedgeplace} fill={"#ff00ff"}/>
				</Chart>

				<Chart id={9} height={400}
					yExtents={[d => d.sighdvertrsiS]}
					origin={(w, h) => [0, h - 530]}
				>
					{/* Yellow */}
					<BarSeries yAccessor={d => d.sigvolestimatwithrsival} fill={"#ffff00"}/>
				</Chart>

				<Chart id={10} height={400}
					yExtents={[d => d.sigsell]}
					origin={(w, h) => [0, h - 530]}
				>
					{/* Brown */}
					<BarSeries yAccessor={d => d.sighdvertrsiS} fill={"#993333"}/>
				</Chart>
				<CrossHairCursor />
			</ChartCanvas>


			
			</div>
		);
	}
}

CandleStickChart.propTypes = {
	data: PropTypes.array.isRequired,
	width: PropTypes.number.isRequired,
	ratio: PropTypes.number.isRequired,
	type: PropTypes.oneOf(["svg", "hybrid"]).isRequired,
};

CandleStickChart.defaultProps = {
	type: "svg",
};
CandleStickChart = fitWidth(CandleStickChart);

export default CandleStickChart;
    