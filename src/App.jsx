import "./App.css";
import model from "./assets/model(8).onnx";

// import * as ort from "onnxruntime-web/dist";
import {
  map,
  path,
  compose,
  curry,
  memoizeWith,
  prop
} from "ramda";
import { useState } from "react";

// const CDN = (asset) => `https://cdn.glitch.global/a3dfaa35-e1fa-40fb-ab69-bbb0261e349f/${asset}`
const MODEL_URL = model;

const getSessionMemoized = memoizeWith(String, function getSession(url) {
  return ort.InferenceSession.create(url);
});

function makeTensorFromString(string) {
  return new ort.Tensor("string", [string], [1, 1]);
}

const categories = {
  9: "Narcotics",
  10: "Protests",
  11: "Organized Crime",
  12: "Corruption",
};
const getItem = curry((obj, x) => obj[x])
const getCategoryById = getItem(categories);

async function runInference(reportTitle, reportContent) {
  try {
    // create a new session and load the specific model.
    const session = await getSessionMemoized(MODEL_URL);

    // prepare inputs. a tensor need its corresponding TypedArray as data
    const [tensorReportTitle, tensorReportContent] = map(makeTensorFromString, [
      reportTitle,
      reportContent,
    ]);

    // prepare feeds. use model input names as keys. These are hardcoded here
    const feeds = {
      CleanTitle: tensorReportTitle,
      CleanReport: tensorReportContent,
    };

    // feed inputs and run
    const results = await session.run(feeds);

    // In our case the output is also hardcoded, called "label"
    const getResultData = path(["label", "data", 0]);

    // Get result data, get the category
    const getCategoryFromResults = compose(getCategoryById, getResultData);

    return getCategoryFromResults(results);
  } catch (e) {
    console.warn(`failed to run ONNX model: ${e}.`);
  }
}

function App() {
  let [prediction, setPrediction] = useState(false);
  const handleSubmit = (e) => {
    const getDomValues = compose(map, prop)("value");
    let { reportTitle, reportContent } = e.target;
    [reportTitle, reportContent] = getDomValues([reportTitle, reportContent]);
    // TODO: clean and lemmatize (or is it even needed?)
    runInference(reportTitle, reportContent).then(setPrediction);
    e.preventDefault();
  };

  const runBatch = async () => {
    const { default: records } = await import("./assets/data.json");
    console.log(`Got ${records.length} records to test`);
    
    // Horrible fast code that calculates this model's accuracy
    let correctCount = 0,
      sameCount = 0;
    for (let record of records) {
      const { CleanTitle, CleanReport, Prediction, Label } = record;
      const jspred = await runInference(CleanTitle, CleanReport);
      // console.log(jspred, Prediction)
      if (jspred == getCategoryById(Label)) correctCount++;
      if (jspred == getCategoryById(Prediction)) sameCount++;
    }
    const pctAccuracy = (count, records) =>
      `${Math.round((count / records.length) * 1e4) / 1e2}%`;

    // And end it with a beautiful side effect
    alert(`Overall accuracy: ${pctAccuracy(correctCount, records)}
    Similarity to Python: ${pctAccuracy(sameCount, records)}`);
  };
  return (
    <>
      <h1 className="title">ONNX Classification</h1>
      <p>No data is sent to the server to run the model.</p>
      <form onSubmit={handleSubmit}>
        <div>
          <input id="reportTitle" placeholder="Report Title" />
        </div>
        <div>
          <textarea id="reportContent" placeholder="Report Content"></textarea>
        </div>
        <div>
          <button type="submit">Guess category</button>
        </div>
      </form>
      <div>I think this is in the {prediction || "..."} category</div>
      <hr />
      <button onClick={runBatch}>Do a model accuracy test</button>
    </>
  );
}

export default App;
