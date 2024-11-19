import { OpenAI } from "openai";
import { config } from "dotenv";
import * as readline from "readline";

config();
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

// This is the example set of documents that would be searched for relevant information
const documents = [
  "The Northern Lights are caused by collisions between electrically charged particles from the sun that enter the Earth's atmosphere.",
  "Quantum entanglement is a phenomenon where particles become interconnected and the state of one can instantaneously affect the state of another, no matter the distance.",
  "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.",
  "The capital of France is Paris, known for its cafes and the Eiffel Tower.",
  "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones.",
];

// Function to create embeddings for documents so that we can represent the documents as vectors in a high dimensional space, this allows us to calculate the similarity between the documents and a hypothetical answer
async function embedDocuments(docs: string[]): Promise<number[][]> {
  const embeddings: number[][] = [];

  for (const doc of docs) {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: doc,
    });
    const [embedding] = response.data;
    embeddings.push(embedding.embedding);
  }

  return embeddings;
}

// Function to calculate cosine similarity between two vectors this measures the similarty between two vectors, in our case documents and a hypothetical answer, the higher the cosine similarity the more similar the two vectors are
// You can envision this as a graph where the angle between the two vectors is measured, the closer the angle is to 0, the more similar the vectors are
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

async function main() {
  console.log("Embedding documents...");
  const documentEmbeddings = await embedDocuments(documents);
  console.log("Document embeddings completed.\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.question("Enter your query: ", async (query: string) => {
    try {
      // Step 1: Generate a hypothetical answer from the users query using and LLM, this gives our RAG system a starting point to work off of
      console.log("\nGenerating hypothetical answer...");
      const hypotheticalResponse = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: query }],
        max_tokens: 150,
        temperature: 0.0,
      });
        let hypotheticalAnswer = "";
        if (hypotheticalResponse.choices[0].message?.content != undefined) {
            hypotheticalAnswer =
            hypotheticalResponse.choices[0].message?.content.trim() ||
            "";
        }
        console.log("Hypothetical Answer:", hypotheticalAnswer, "\n");
        // Step 2: Create embedding for the hypothetical answer these values are generated from an extermal embedding algorithm, in this case we are using the text-embedding-3-small model
        // These embedding values will be passed to the cosine similarity function to calculate the similarity between the hypothetical answer and the documents
      console.log("Embedding hypothetical answer...");
      const embeddingResponse = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: hypotheticalAnswer,
      });
      const [hypotheticalEmbeddingData] = embeddingResponse.data;
      const hypotheticalEmbedding = hypotheticalEmbeddingData.embedding;
      // Step 3: Compute the cosine similarities with document embeddings
      console.log("Calculating similarities...");
      const similarities = documentEmbeddings.map((docEmbedding, index) => {
        const similarity = cosineSimilarity(
          hypotheticalEmbedding,
          docEmbedding
        );
        return { index, similarity };
      });
      // Step 4: Sort documents by similarity
      similarities.sort((a, b) => b.similarity - a.similarity);
      // Step 5: Retrieve top K similar documents
      const topK = 2; // This is the number of documents that will be used to generate the final answer in our case we are just taking the best two documents
      const topDocuments = similarities
        .slice(0, topK)
        .map((sim) => documents[sim.index]);

      console.log("\nTop relevant documents:");
      topDocuments.forEach((doc, idx) => {
        console.log(`${idx + 1}. ${doc}`);
      });
      console.log("\nGenerating final answer using retrieved documents...");
      const finalPrompt = `
Using the information below, answer the question.

Information:
${topDocuments.join("\n")}

Question:
${query}

Answer:
    `;
    const finalResponse = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [{ role: "user", content: finalPrompt }],
    max_tokens: 150,
    temperature: 0.7,
    });
    let finalAnswer = "";
    if (finalResponse.choices[0].message?.content != undefined) {
        finalAnswer =
        finalResponse.choices[0].message?.content.trim() || "";
    }
      console.log("\nFinal Answer:", finalAnswer);
    } catch (error) {
      console.error("An error occurred:", error);
    } finally {
      rl.close();
    }
  });
}

main();