import React, { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend, annotationPlugin);

const introFeatures = [
  { name: "dep_pair_conj_PROPN", value: 0.09, mean: 0.05, weight: +0.98 },
  { name: "dep_pair_amod_PROPN", value: 0.18, mean: 0.13, weight: +0.42 },
  { name: "dep_pair_nmod_NOUN", value: 0.12, mean: 0.1, weight: +0.38 },
  { name: "dep_pair_aux_VERB", value: 0.08, mean: 0.06, weight: -0.31 },
  { name: "dep_pair_nummod_PROPN", value: 0.05, mean: 0.07, weight: -0.29 },
  { name: "adv_ratio", value: 0.06, mean: 0.04, weight: +0.25 },
  { name: "passive_voice_ratio", value: 0.15, mean: 0.1, weight: -0.22 },
  { name: "present_ratio", value: 0.35, mean: 0.4, weight: +0.18 },
  { name: "polarity", value: 0.72, mean: 0.75, weight: +0.14 },
  { name: "dep_pair_prep_VERB", value: 0.09, mean: 0.08, weight: +0.10 },
  { name: "dep_pair_nsubj_VERB", value: 0.11, mean: 0.12, weight: +0.08 },
  { name: "promo_word_ratio", value: 0.06, mean: 0.05, weight: +0.05 },
  { name: "dep_pair_appos_PROPN", value: 0.04, mean: 0.03, weight: +0.02 }
];

const WritingStyleEvaluator = () => {
  const [text, setText] = useState("");
  const [score, setScore] = useState(null);
  const [percentile, setPercentile] = useState(null);

  const handleEvaluate = () => {
    setScore(0.76);
    setPercentile(82);
  };

  const mockDistribution = {
    labels: Array.from({ length: 10 }, (_, i) => `${i * 10}-${i * 10 + 10}%`),
    datasets: [
      {
        label: 'Distribution',
        data: [1, 4, 9, 15, 18, 20, 15, 10, 5, 3],
        backgroundColor: '#4E2A84'
      }
    ]
  };

  const mockOptions = {
    plugins: {
      annotation: {
        annotations: {
          line1: {
            type: 'line',
            xMin: 8.2,
            xMax: 8.2,
            borderColor: 'red',
            borderWidth: 2,
            label: {
              content: 'You are here',
              enabled: true,
              position: 'end'
            }
          }
        }
      }
    },
    scales: {
      x: { title: { display: true, text: 'Percentile' } },
      y: { title: { display: true, text: 'Count' } }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-10 flex flex-col items-center justify-center">
      <div className="bg-white p-8 rounded-xl shadow-md w-full max-w-2xl">
        <h1 className="text-2xl font-bold text-center text-gray-800 mb-6">AcademicGPT: Scientific Writing Evaluator</h1>
        <textarea
          className="w-full h-40 p-4 border border-gray-300 rounded mb-4 shadow-sm focus:ring-2 focus:ring-blue-400"
          placeholder="Paste your Introduction here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          className="w-full py-2 bg-blue-600 text-white font-semibold rounded hover:bg-blue-700"
          onClick={handleEvaluate}
        >
          ✨ Evaluate
        </button>
      </div>

      {score && (
        <div className="bg-white p-8 rounded-xl shadow-md w-full max-w-2xl mt-8">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Your Score: <span className="text-blue-600">{score}</span> — Top <span className="text-green-600">{percentile}%</span></h2>
          <div className="mb-6">
            <Bar data={mockDistribution} options={mockOptions} />
          </div>
          <h3 className="text-md font-semibold text-gray-700 mb-2">Feature-level Analysis</h3>
          <ul className="space-y-3 text-sm text-gray-600">
            {introFeatures.map((f, i) => (
              <li key={i} className="border-b pb-2">
                <span className="font-medium text-gray-800">{f.name}</span><br />
                value: <span className="text-blue-600">{f.value}</span>, mean: <span className="text-purple-600">{f.mean}</span>, weight: <span className={f.weight > 0 ? "text-green-600" : "text-red-600"}>{f.weight > 0 ? '+' : ''}{f.weight.toFixed(2)}</span><br />
                <span className="italic text-gray-500">Explanation: [Add explanation for {f.name}]</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default WritingStyleEvaluator;