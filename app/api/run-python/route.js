import { spawn } from 'child_process';
import { NextResponse } from 'next/server';

export async function POST(req) {
  try {
    const { query } = await req.json();

    const pythonProcess = spawn('python', ['perform_rag.py', query]);

    return new Promise((resolve, reject) => {
      let responseText = '';

      pythonProcess.stdout.on('data', (data) => {
        responseText += data.toString();
      });

      pythonProcess.stdout.on('end', () => {
        console.log("Python script output:", responseText); // Debug print
        resolve(NextResponse.json({ response: responseText }));
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error('Error:', data.toString());
        reject(NextResponse.json({ error: 'Error generating response. Please try again later.' }));
      });
    });
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json({ error: 'Error generating response. Please try again later.' });
  }
}
