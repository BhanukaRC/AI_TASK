import { answerQuery, AnswerQueryResponse } from './index.mts';
import { qaQuestions, TestCase } from './qa_questions.mts';

const testCases: TestCase[] = qaQuestions;

let passed = 0;
let failed = 0;

(async () => {
  for (const [i, test] of testCases.entries()) {
    console.log(`\nTest #${i + 1}: ${test.question || 'This is an empty question'}`);
    const { answer, confidence }: AnswerQueryResponse = await answerQuery(test.question);
    console.log(`Answer: ${answer}`);
    console.log(`Confidence: ${confidence}`);
    let pass = true;
    if (typeof test.expectedAnswer === 'string') {
      if (!answer.includes(test.expectedAnswer)) {
        console.log(`  Expected answer to include: ${test.expectedAnswer}`);
        pass = false;
      }
    }
    if (Array.isArray(test.expectedKeywords)) {
      for (const kw of test.expectedKeywords) {
        if (!answer.toLowerCase().includes(kw.toLowerCase())) {
          console.log(`  Missing keyword: ${kw}`);
          pass = false;
        }
      }
    }
    if (typeof test.minConfidence === 'number' && confidence < test.minConfidence) {
      console.log(`  Confidence ${confidence} < min required ${test.minConfidence}`);
      pass = false;
    }
    if (pass) {
      console.log('PASS');
      passed++;
    } else {
      console.log('FAIL');
      failed++;
    }
    // To avoid rate limiting, wait 2.5 seconds between questions
    if (i < testCases.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 2500));
    }
  }
  console.log(`\nTest summary: ${passed} passed, ${failed} failed.`);
})();
