import { useState, useEffect, useRef } from 'react';
import { MessageCircle, X, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState([
    {
      text: "👋 Hi! I'm Amirlahi's AI assistant. I can help you learn more about his work, projects, or answer any questions you might have! 🚀",
      isBot: true
    }
  ]);
  const [inputValue, setInputValue] = useState('');

  const API_BASE_URL = 'http://localhost:8000/api';

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue;
    const newMessages = [
      ...messages,
      { text: userMessage, isBot: false }
    ];

    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the FastAPI backend
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          conversation_id: conversationId
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      // Update conversation ID for multi-turn conversations
      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      // Add bot response
      setMessages(prev => [...prev, { text: data.response, isBot: true }]);
    } catch (error) {
      console.error('Error fetching chatbot response:', error);
      // Fallback error message
      setMessages(prev => [...prev, {
        text: "I apologize, but I'm having trouble connecting right now. Please try again later or contact Amirlahi directly through the contact form.",
        isBot: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <>
      {/* Chat Window */}
      {isOpen && (
        <Card className="fixed bottom-24 right-6 w-80 max-w-[calc(100vw-3rem)] gradient-border bg-background-secondary border-0 shadow-elegant z-50 animate-slide-in-right">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg gradient-text">Chat with Amirlahi</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsOpen(false)}
                className="h-8 w-8 p-0 hover:bg-background-tertiary"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-4">
            {/* Messages */}
            <div className="h-60 overflow-y-auto space-y-3 pr-2">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.isBot ? 'justify-start' : 'justify-end'}`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-lg text-sm ${
                      message.isBot
                        ? 'bg-background-tertiary text-foreground'
                        : 'bg-gradient-primary text-background'
                    }`}
                  >
                    {message.text}
                  </div>
                </div>
              ))}
              
              {/* Loading indicator */}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-background-tertiary text-foreground max-w-[80%] p-3 rounded-lg text-sm">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-accent-primary rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-accent-primary rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                        <div className="w-2 h-2 bg-accent-primary rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                      </div>
                      <span className="text-xs text-foreground-muted">Typing...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
            
            {/* Input */}
            <div className="flex gap-2">
              <Input
                placeholder="Type your message..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                className="flex-1 bg-background border-border focus:border-accent-primary relative z-10"
              />
              <Button
                onClick={handleSendMessage}
                disabled={isLoading}
                className="btn-gradient text-background px-3"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Chat Button */}
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full btn-gradient text-background shadow-elegant z-50 animate-glow-pulse"
        size="lg"
      >
        {isOpen ? <X className="w-6 h-6" /> : <MessageCircle className="w-6 h-6" />}
      </Button>
    </>
  );
};

export default Chatbot;