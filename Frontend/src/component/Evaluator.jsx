import React, { useState, useEffect } from "react";
import "./Evaluator.css";
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
  Title
} from 'chart.js';

import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend, Title, annotationPlugin);

const introFeatures = [
  {
    name: "INTRODUCTION_sentence_length_characters_max",
    explanation: "Long sentences may affect readability and make comprehension harder for readers."
  },
  {
    name: "INTRODUCTION_dep_pair_appos_PROPN",
    explanation: "Proper noun appositions can enhance specificity but may also introduce complexity."
  },
  {
    name: "INTRODUCTION_dep_pair_amod_NOUN",
    explanation: "Adjective-noun modifiers reflect descriptive richness, but excessive use can reduce clarity."
  },
  {
    name: "INTRODUCTION_promo_word_ratio",
    explanation: "Promotional words may undermine the objectivity expected in scientific writing."
  },
  {
    name: "INTRODUCTION_pp_as_postmodifier_count",
    explanation: "Frequent postmodifying prepositional phrases may contribute to syntactic complexity."
  },
  {
    name: "INTRODUCTION_topic_similarity_to_conclusion_mean",
    explanation: "Measures coherence; higher similarity suggests consistent messaging across the paper."
  },
  {
    name: "INTRODUCTION_perfect_ratio",
    explanation: "Perfect tense usage indicates completeness but should be balanced to avoid verbosity."
  }
];

const mockFeatureValuesIntro = {
  "INTRODUCTION_sentence_length_characters_max": 204,
  "INTRODUCTION_dep_pair_appos_PROPN": 4,
  "INTRODUCTION_dep_pair_amod_NOUN": 11,
  "INTRODUCTION_promo_word_ratio": 0.015,
  "INTRODUCTION_pp_as_postmodifier_count": 6,
  "INTRODUCTION_topic_similarity_to_conclusion_mean": 0.72,
  "INTRODUCTION_perfect_ratio": 0.09,
};

const featureDisplayNamesIntro = {
  "INTRODUCTION_sentence_length_characters_max": "Maximum Sentence Length (Characters)",
  "INTRODUCTION_dep_pair_appos_PROPN": "Proper Noun Apposition",
  "INTRODUCTION_dep_pair_amod_NOUN": "Adjective-Noun Modifier",
  "INTRODUCTION_promo_word_ratio": "Promotional Word Ratio",
  "INTRODUCTION_pp_as_postmodifier_count": "Prepositional Phrase as Noun Modifier",
  "INTRODUCTION_topic_similarity_to_conclusion_mean": "Topic Similarity to Conclusion",
  "INTRODUCTION_perfect_ratio": "Perfect Tense Usage Ratio"
};

const conclusionFeatures = [
  {
    name: "CONCLUSION_dale_chall_readability_score",
    explanation: "Reflects ease of reading; lower scores indicate more accessible writing."
  },
  {
    name: "CONCLUSION_dep_pair_compound_PROPN",
    explanation: "Compound proper nouns can offer specificity but may require contextual clarity."
  },
  {
    name: "CONCLUSION_CP_T",
    explanation: "Complex phrases per T-unit capture sentence complexity and syntactic depth."
  },
  {
    name: "CONCLUSION_num_citations_in_sentence_mean",
    explanation: "Frequent citations demonstrate grounding in literature but may impact flow."
  },
  {
    name: "CONCLUSION_dep_pair_det_NOUN",
    explanation: "Determiner-noun constructions reflect syntactic precision and clarity."
  },
  {
    name: "CONCLUSION_dep_pair_amod_NOUN",
    explanation: "Adjective-noun modifiers enhance descriptive detail but may complicate parsing."
  },
  {
    name: "CONCLUSION_past_ratio",
    explanation: "Past tense is often used in describing results; excessive use may dull immediacy."
  }
];

const mockFeatureValuesConclusion = {
  "CONCLUSION_dale_chall_readability_score": 7.9,
  "CONCLUSION_dep_pair_compound_PROPN": 3,
  "CONCLUSION_CP_T": 1.15,
  "CONCLUSION_num_citations_in_sentence_mean": 1.8,
  "CONCLUSION_dep_pair_det_NOUN": 9,
  "CONCLUSION_dep_pair_amod_NOUN": 6,
  "CONCLUSION_past_ratio": 0.28,
};

const featureDisplayNamesConclusion = {
  "CONCLUSION_dale_chall_readability_score": "Writing Score (Dale-Chall Readability)",
  "CONCLUSION_dep_pair_compound_PROPN": "Compound Proper Noun",
  "CONCLUSION_CP_T": "Complex Phrases per T-unit",
  "CONCLUSION_num_citations_in_sentence_mean": "Number of Citations",
  "CONCLUSION_dep_pair_det_NOUN": "Determiner-Noun Construction",
  "CONCLUSION_dep_pair_amod_NOUN": "Adjective-Noun Modifier",
  "CONCLUSION_past_ratio": "Verb Tense (Past)"
};

const Evaluator = () => {
  const [text, setText] = useState("");
  const [wordCount, setWordCount] = useState(0);
  const [score, setScore] = useState(null);
  const [percentile, setPercentile] = useState(null);
  const [featureDistributions, setFeatureDistributions] = useState(null);
  const [openFeatureIndex, setOpenFeatureIndex] = useState(null);
  const [selectedSection, setSelectedSection] = useState("Introduction");

  useEffect(() => {
    fetch("http://localhost:5000/feature-distribution")
      .then((res) => res.json())
      .then((data) => setFeatureDistributions(data));  // 
  }, []);
  

  const handleTextChange = (e) => {
    const txt = e.target.value;
    setText(txt);
    const count = txt.trim().split(/\s+/).filter(w => w).length;
    setWordCount(count);
  };

  const handleEvaluate = () => {
    setScore(0.76);
    setPercentile(82);
  };

  const currentFeatures =
    selectedSection === "Introduction"
      ? introFeatures
      : selectedSection === "Conclusion"
      ? conclusionFeatures
      : [];

  const currentFeatureValues =
    selectedSection === "Introduction"
      ? mockFeatureValuesIntro
      : selectedSection === "Conclusion"
      ? mockFeatureValuesConclusion
      : {};

  const currentDisplayNames =
    selectedSection === "Introduction"
      ? featureDisplayNamesIntro
      : selectedSection === "Conclusion"
      ? featureDisplayNamesConclusion
      : {};

  return (
    <div>
      <div className="banner">
        🎉 <strong>Welcome Onboard!</strong>
        <button className="close-button">Close</button>
      </div>
      <div className="container">
        <div className="left-panel">
          <div className="section-selector">
            <p>Select the section of the paper you are evaluating</p>
            <select className="section-select"
                    value={selectedSection}
                    onChange={(e) => setSelectedSection(e.target.value)}
                    >
              <option>Abstract</option>
              <option>Introduction</option>
              <option>Background</option>
              <option>Methods</option>
              <option>Results</option>
              <option>Conclusion</option>
            </select>
          </div>
          <div className="editor-box">
            <div className="word-count">{wordCount} words</div>
            <textarea
              value={text}
              onChange={handleTextChange}
              placeholder="Type or paste your text to receive language suggestions or type '/' to explore AI features"
            />
            <button className="evaluate-button" onClick={handleEvaluate}>Evaluate</button>
          </div>
        </div>

        <div className="right-panel">
          <div className="sidebar-title">✏️ Analysis</div>
          {score !== null && (
            <>

              {featureDistributions && currentFeatures.length > 0 && (
                <div className="sidebar-section">
                  <h3>📉 Feature Distributions</h3>
                  {currentFeatures.map((f, idx) => {
                    const feature = f.name;
                    const values = featureDistributions[feature];
                    if (!values) return null;

                    const bins = Array(10).fill(0);
                    const min = Math.min(...values);
                    const max = Math.max(...values);
                    const binSize = (max - min) / 10;

                    const binLabels = bins.map((_, i) =>
                      `${(min + i * binSize).toFixed(2)}-${(min + (i + 1) * binSize).toFixed(2)}`
                    );

                    values.forEach(val => {
                      const index = Math.min(9, Math.floor((val - min) / binSize));
                      bins[index]++;
                    });

                    const chartData = {
                      labels: bins.map((_, i) =>
                        `${(min + i * binSize).toFixed(2)}-${(min + (i + 1) * binSize).toFixed(2)}`
                      ),
                      datasets: [{
                        label: feature,
                        data: bins,
                        backgroundColor: '#836EAA'
                      }]
                    };

                    // Get the user's value for this feature
                    const userValue = currentFeatureValues[feature];
                    
                    // Calculate which bin the user's value falls into
                    const userBinIndex = userValue !== undefined 
                      ? Math.min(9, Math.floor((userValue - min) / binSize))
                      : null;

                    const chartOptions = {
                      responsive: true,
                      title: {
                        display: true,
                        font: {
                          size: 13,
                        }
                      },
                      plugins: {
                        legend: { display: false },
                        annotation: {
                          annotations: userValue !== undefined ? {
                            line1: {
                              type: 'line',
                              scaleID: 'x',
                              value: userBinIndex !== null ? userBinIndex : 0,
                              borderColor: 'red',
                              borderWidth: 2,
                              label: {
                                display: true,
                                content: `Your feature value: ${userValue}`,
                                position: 'start',
                                backgroundColor: 'rgba(255, 0, 0, 0.7)',
                                color: 'white',
                                font: {
                                  weight: 'bold'
                                }
                              }
                            }
                          } : {}
                        }
                      },
                      scales: {
                        x: {
                          title: {
                            display: true,
                            text: currentDisplayNames[feature] || feature, // x-axis label
                            font: { size: 14, weight: 'bold' }
                          }
                        },
                        y: {
                          title: {
                            display: true,
                            text: 'Paper Count', // y-axis label (optional)
                            font: { size: 14, weight: 'bold' }
                          }
                        }
                      }
                    };

                    return (
                      <div key={idx} style={{ marginBottom: "30px" }}>
                        <h4>{currentDisplayNames[feature] || feature}</h4>
                        <Bar data={chartData} options={chartOptions} />
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="sidebar-section">
                <h3>Feature-level Analysis</h3>
                <ul className="feature-list">
                  {currentFeatures.map((f, i) => (
                    <li key={i} className="feature-item">
                      <div className="feature-header" onClick={() => setOpenFeatureIndex(openFeatureIndex === i ? null : i)}>
                        <span>{f.name}</span>
                        <span className="chevron">{openFeatureIndex === i ? '▾' : '▸'}</span>
                      </div>
                      {openFeatureIndex === i && (
                        <div className="feature-explanation">{f.explanation}</div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}

          {score === null && (
            <div className="sidebar-section">
              <button className="how-it-works">HOW IT WORKS</button>
              <p className="description">Accept or reject language suggestions to improve the quality of your text. Suggestions will appear once you start writing or open a document.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Evaluator;
