import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState } from "react";
import Auth from "./components/Auth";
import ClientPage from "./components/ClientPage";
import AdminPage from "./components/AdminPage";
import Navbar from "./components/Navbar";


function App() {
  const [user, setUser] = useState(null);

  return (
    <Router>
      <Navbar user={user} setUser={setUser} />
      <Routes>
        <Route path="/" element={<Auth setUser={setUser} />} />
        <Route path="/client" element={user && !user.is_admin ? <ClientPage user={user} /> : <Auth setUser={setUser} />} />
        <Route path="/admin" element={user && user.is_admin ? <AdminPage user={user} /> : <Auth setUser={setUser} />} />
      </Routes>
    </Router>
  );
}

export default App;