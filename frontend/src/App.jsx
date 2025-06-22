import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "./services/api";
import Auth from "./components/Auth";
import AdminPage from "./components/AdminPage";
import ClientPage from "./components/ClientPage";
import Navbar from "./components/Navbar";

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true); // To prevent blank screen flash

  useEffect(() => {
    const checkSession = async () => {
      try {
        const res = await axios.get("/session");
        if (res.data.user) {
          setUser(res.data.user);
        }
      } catch (err) {
        console.error("No session found or error occurred");
      } finally {
        setLoading(false); // ✅ Always set loading to false
      }
    };

    checkSession();
  }, []);

  if (loading) return <div className="p-4 text-center">Loading...</div>; // Show spinner or message

  return (
    <Router>
      {user && <Navbar user={user} setUser={setUser} />}
      <Routes>
        <Route path="/" element={<Auth setUser={setUser} />} />
        <Route path="/client" element={user && !user.is_admin ? <ClientPage user={user} /> : <Auth setUser={setUser} />} />
        <Route path="/admin" element={user?.is_admin ? <AdminPage user={user} /> : <Auth setUser={setUser} />} />
      </Routes>
    </Router>
  );
}

export default App;
