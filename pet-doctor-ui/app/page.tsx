import React from 'react';
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

// ==========================================
// Main Page Component
// ==========================================
export default function PetDoctorChatUI() {
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

          {MOCK_MESSAGES.map((message) => (
            <div
              key={message.id}
              className={`${styles.messageRow} ${message.role === 'user' ? styles.userRow : ''}`}
            >
              <div className={`${styles.messageContent} ${message.role === 'user' ? styles.userMessageContent : ''}`}>
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
                  className={`${styles.bubble} ${message.role === 'user' ? styles.userBubble : styles.assistantBubble}`}
                >
                  <p className={styles.bubbleText}>
                    {message.content.split('**').map((part, index) =>
                      index % 2 === 1 ? <strong key={index}>{part}</strong> : part,
                    )}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </main>

      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <div className={styles.inputWrap}>
            <textarea
              rows={1}
              placeholder="Describe your dog's symptoms..."
              className={styles.textarea}
            />

            <button
              type="button"
              className={styles.sendButton}
            >
              <Send size={18} className={styles.sendIcon} strokeWidth={2.5} />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
