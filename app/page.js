'use client'

import { Box, Button, Stack, TextField } from "@mui/material";
import { useState } from "react";

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: 'user',
      parts: [{ text: `Hi, I'm a Support Agent, how can I assist you today?` }],
      isModel: true, // Add a flag to indicate this message should be displayed as from the model
    }
  ]);

  const [message, setMessage] = useState('');

  const sendMessage = async () => {
    setMessage('');
    setMessages((prevMessages) => [
      ...prevMessages,
      { role: "user", parts: [{ text: message }] },
      { role: "model", parts: [{ text: "" }] },
    ]);

    try {
      const response = await fetch('/api/run-python', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: message }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      if (data.response) {
        setMessages((prevMessages) => [
          ...prevMessages,
          { role: "model", parts: [{ text: data.response }] }
        ]);
      } else {
        setMessages((prevMessages) => [
          ...prevMessages,
          { role: "model", parts: [{ text: 'No response from Python script.' }] }
        ]);
      }
    } catch (error) {
      console.error('Error generating response:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: "model", parts: [{ text: 'Error generating response. Please try again later.' }] }
      ]);
    }
  }

  return (
    <Box 
      width="100vw" 
      height="100vw"
      display="flex"
      flexDirection="column"
      justifyContent="center"
      alignItems="center"
    >
      <Stack direction="column" width="600px" height="700px" border="1px solid black" p={2} spacing={3}>
        <Stack direction="column" spacing={2} flexGrow={1} overflow="auto" maxHeight="100%">
          {
            messages.map((message, index) => {
              return (
                <Box key={index} display='flex' justifyContent={
                  message.isModel || message.role === 'model' ? 'flex-start' : 'flex-end'
                }>
                  <Box bgcolor={
                    message.isModel || message.role === 'model'
                      ? 'primary.main'
                      : 'secondary.main' 
                    }
                    color="white"
                    borderRadius={16}
                    p={3}>
                      {message.parts[0].text}
                    </Box>
                </Box>
              )
            })
          }
        </Stack>
        <Stack direction="row" spacing={2}>
          <TextField 
            label="message" 
            fullWidth 
            value={message} 
            onChange={(e) => setMessage(e.target.value)} 
          />
          <Button variant="contained" onClick={sendMessage}>Send</Button>
        </Stack>
      </Stack>
    </Box>
  )
}
