import { useState } from 'react';
import { Mail, Phone, MapPin, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';

const ContactSection = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const { toast } = useToast();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Message Sent!",
      description: "Thank you for your message. I'll get back to you soon.",
    });
    setFormData({ name: '', email: '', subject: '', message: '' });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  return (
    <section id="contact" className="py-20 bg-background-secondary/50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 scroll-reveal">
          <h2 className="text-4xl font-bold mb-4">
            Get In <span className="gradient-text">Touch</span>
          </h2>
          <div className="w-24 h-1 bg-gradient-primary mx-auto rounded-full mb-6"></div>
          <p className="text-xl text-foreground-muted max-w-2xl mx-auto">
            Have a project in mind or want to discuss data science opportunities? 
            I'd love to hear from you!
          </p>
        </div>
        
        <div className="grid lg:grid-cols-2 gap-12">
          {/* Contact Info */}
          <div className="space-y-8 scroll-reveal">
            <div>
              <h3 className="text-2xl font-semibold mb-6 gradient-text">
                🤝 Let's Connect
              </h3>
              <p className="text-foreground-muted leading-relaxed mb-8">
                I'm always interested in discussing new opportunities, 
                collaborating on innovative projects, or simply chatting about 
                the latest developments in data science and AI.
              </p>
            </div>
            
            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-accent-primary/10 rounded-lg flex items-center justify-center">
                  <Mail className="w-6 h-6 text-accent-primary" />
                </div>
                <div>
                  <h4 className="font-semibold">Email</h4>
                  <p className="text-foreground-muted">adescofaj@gmail.com</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-accent-primary/10 rounded-lg flex items-center justify-center">
                  <Phone className="w-6 h-6 text-accent-primary" />
                </div>
                <div>
                  <h4 className="font-semibold">Phone</h4>
                  <p className="text-foreground-muted">07765624800</p>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-accent-primary/10 rounded-lg flex items-center justify-center">
                  <MapPin className="w-6 h-6 text-accent-primary" />
                </div>
                <div>
                  <h4 className="font-semibold">Location</h4>
                  <p className="text-foreground-muted">Bournemouth, United Kingdom</p>
                </div>
              </div>
            </div>
          </div>
          
          {/* Contact Form */}
          <Card className="gradient-border bg-background-secondary border-0 scroll-reveal">
            <CardHeader>
              <CardTitle className="text-accent-primary">💬 Send Message</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid sm:grid-cols-2 gap-4">
                  <div>
                    <Input
                      name="name"
                      placeholder="Your Name"
                      value={formData.name}
                      onChange={handleChange}
                      className="bg-background border-border focus:border-accent-primary relative z-10"
                      required
                      autoComplete="name"
                    />
                  </div>
                  <div>
                    <Input
                      name="email"
                      type="email"
                      placeholder="Your Email"
                      value={formData.email}
                      onChange={handleChange}
                      className="bg-background border-border focus:border-accent-primary relative z-10"
                      required
                      autoComplete="email"
                    />
                  </div>
                </div>
                
                <Input
                  name="subject"
                  placeholder="Subject"
                  value={formData.subject}
                  onChange={handleChange}
                  className="bg-background border-border focus:border-accent-primary relative z-10"
                  required
                />
                
                <Textarea
                  name="message"
                  placeholder="Your Message"
                  rows={5}
                  value={formData.message}
                  onChange={handleChange}
                  className="bg-background border-border focus:border-accent-primary resize-none relative z-10"
                  required
                />
                
                <Button 
                  type="submit" 
                  className="btn-gradient text-background font-semibold w-full"
                  size="lg"
                >
                  <Send className="w-5 h-5 mr-2" />
                  Send Message
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default ContactSection;