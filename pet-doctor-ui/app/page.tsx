'use client';

import React, { useState } from 'react';
import { Send, PawPrint, Stethoscope, User, ChevronLeft, MoreHorizontal } from 'lucide-react';
import styles from './page.module.css';

// ==========================================
// Mock Data (For simulating the conversation)
// ==========================================

const MOCK_MESSAGES = [
  {
    id: '1',
    role: 'assistant',
    content: 'Hello! I am your AI Pet Doctor Assistant. While I cannot replace a real veterinarian\'s physical examination, I can provide advice based on veterinary medicine data.\n\nWhat seems to be the problem with your dog?',
  },
  {
    id: '2',
    role: 'user',
    content: 'My dog hasn\'t been eating for two days, has diarrhea, and seems very lethargic.',
  },
  {
    id: '3',
    role: 'assistant',
    content: 'I understand your dog is experiencing a loss of appetite, diarrhea, and lethargy. These are common symptoms, but they can stem from various causes.\n\n**Based on statistical data and common cases, potential reasons include:**\n1. **Dietary Indiscretion**: Eating garbage, sudden diet changes, or spoiled food.\n2. **Parasitic Infection**: Such as roundworms or giardia.\n3. **Viral Infection**: If your dog is not fully vaccinated, we must rule out Parvovirus.\n4. **Stress or Anxiety**: Triggered by recent environmental changes.\n\nI recommend checking your dog\'s temperature first and ensuring they have access to fresh water to prevent dehydration. If symptoms persist for over 24 hours, or if you notice vomiting or bloody stools, please take them to a vet clinic immediately for bloodwork and a fecal exam.',
  },
  {
    id: '4',
    role: 'user',
    content: 'Got it. I just took his temperature and it is 101.5°F (38.6°C). He is drinking water but still refusing food. Should we go to the hospital right now?',
  },
  {
    id: '5',
    role: 'assistant',
    content: 'A dog\'s normal temperature ranges from 101.0°F to 102.5°F (38.3°C to 39.2°C). So, 101.5°F is perfectly normal, which is good news and rules out a high fever for now.\n\nGiven the normal temperature but continued fasting and diarrhea, **I recommend the following conservative measures and close observation for the next 12-24 hours:**\n\n* **Fast but provide water**: Withhold food for 12 hours to let the gastrointestinal tract rest, but ensure constant access to clean water.\n* **Monitor lethargy**: If he begins hiding, shows extreme weakness, or starts vomiting repeatedly, do not wait—seek immediate veterinary care.\n\nIf he shows hunger after 12 hours, try offering a very small amount of a bland diet (like boiled chicken breast and white rice). If he vomits after eating or the diarrhea worsens, go to the clinic.',
  },
];

type Message = {
  id: string;
  role: 'assistant' | 'user';
  content: string;
};

// ==========================================
// Main Page Component
// ==========================================
export default function PetDoctorChatUI() {

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        "Hello! I am your AI Pet Doctor Assistant. While I cannot replace a real veterinarian's physical examination, I can provide advice based on veterinary medicine data.\n\nWhat seems to be the problem with your dog?",
    },
  ]);

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        let errorText = 'Request failed';
        try {
          const errData = await res.json();
          errorText = errData.detail || errorText;
        } catch {
          errorText = `HTTP ${res.status}`;
        }
        throw new Error(errorText);
      }

      const data: { answer: string } = await res.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content:
          error instanceof Error
            ? `Sorry, request failed: ${error.message}`
            : 'Sorry, request failed.',
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <button className={styles.iconButton}>
          <ChevronLeft size={24} strokeWidth={2} />
        </button>

        <div className={styles.titleBlock}>
          <div className={styles.titleRow}>
            <Stethoscope size={18} className={styles.brandIcon} strokeWidth={2.5} />
            <h1 className={styles.title}>Pet Doctor AI</h1>
          </div>
          <span className={styles.subtitle}>Online Assistant</span>
        </div>

        <button className={styles.iconButton}>
          <MoreHorizontal size={24} strokeWidth={2} />
        </button>
      </header>

      <main className={styles.main}>
        <div className={styles.messages}>
          <div className={styles.disclaimerWrap}>
            <span className={styles.disclaimer}>
              AI advice is for reference only and cannot replace physical veterinary care.
            </span>
          </div>

          {messages.map((message) => (
            <div
              key={message.id}
              className={`${styles.messageRow} ${message.role === 'user' ? styles.userRow : ''}`}
            >
              <div
                className={`${styles.messageContent} ${
                  message.role === 'user' ? styles.userMessageContent : ''
                }`}
              >
                {message.role === 'assistant' ? (
                  <div className={styles.assistantAvatar}>
                    <PawPrint size={16} className={styles.brandIcon} strokeWidth={2.5} />
                  </div>
                ) : (
                  <div className={styles.userAvatar}>
                    <User size={16} className={styles.userAvatarIcon} strokeWidth={2.5} />
                  </div>
                )}

                <div
                  className={`${styles.bubble} ${
                    message.role === 'user' ? styles.userBubble : styles.assistantBubble
                  }`}
                >
                  <p className={styles.bubbleText} style={{ whiteSpace: 'pre-wrap' }}>
                    {message.content}
                  </p>
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className={styles.messageRow}>
              <div className={styles.messageContent}>
                <div className={styles.assistantAvatar}>
                  <PawPrint size={16} className={styles.brandIcon} strokeWidth={2.5} />
                </div>
                <div className={`${styles.bubble} ${styles.assistantBubble}`}>
                  <p className={styles.bubbleText}>Thinking...</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <div className={styles.inputWrap}>
            <textarea
              rows={1}
              placeholder="Describe your dog's symptoms..."
              className={styles.textarea}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />

            <button
              type="button"
              className={styles.sendButton}
              onClick={handleSend}
              disabled={loading}
            >
              <Send size={18} className={styles.sendIcon} strokeWidth={2.5} />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
