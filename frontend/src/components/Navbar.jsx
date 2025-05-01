import React from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "../services/api";

const Navbar = ({ user, setUser }) => {
    const navigate = useNavigate();

    const handleLogout = async () => {
        try {
            await axios.post("/logout");
            setUser(null);
            navigate("/");
        } catch (error) {
            console.error("Logout failed");
        }
    };

    return (
        <nav className="flex items-center justify-between px-6 py-3 bg-[#1b1b1b] text-white shadow-md">
            <div className="text-xl font-bold">DENGFL</div>
            <div className="flex gap-5 items-center">
                <Link className="hover:underline" to="/client">Client Page</Link>
                <Link className="hover:underline" to="/admin">Admin Page</Link>

                {user && (
                    <>
                        <span className="text-sm italic">Logged in as: {user.name}</span>
                        <button onClick={handleLogout} className="ml-4 bg-red-500 px-3 py-1 rounded text-sm">
                            Logout
                        </button>
                    </>
                )}
            </div>
        </nav>
    );
};

export default Navbar;
