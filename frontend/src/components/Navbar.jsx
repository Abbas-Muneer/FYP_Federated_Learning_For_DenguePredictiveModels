import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "../services/api";

const Navbar = ({ user, setUser }) => {
    const navigate = useNavigate();
    const location = useLocation();

    const handleLogout = async () => {
        try {
            await axios.post("/logout");
            setUser(null);
            navigate("/");
        } catch (error) {
            console.error("Logout failed");
        }
    };

    const handleRoleSwitch = async (targetPath) => {
        try {
            await axios.post("/logout");
            setUser(null);
            navigate(targetPath); 
        } catch (error) {
            console.error("Role switch failed");
        }
    };

    return (
        <nav className="bg-[#1b1b1b] text-white p-4 flex justify-between items-center">
            <div className="text-lg font-bold">FL System</div>
            <div className="flex gap-4 items-center">
                <button>Home</button>
                <button>FL Train</button>
                {user?.is_admin && location.pathname !== "/client" && (
                    <button
                        onClick={() => handleRoleSwitch("/client")}
                        className="hover:underline"
                    >
                        Client Page
                    </button>
                )}
                {!user?.is_admin && location.pathname !== "/admin" && (
                    <button
                        onClick={() => handleRoleSwitch("/admin")}
                        className="hover:underline"
                    >
                        Admin Page
                    </button>
                )}
                {user && (
                    <button onClick={handleLogout} className="hover:underline">
                        Logout
                    </button>
                )}
            </div>
        </nav>
    );
};

export default Navbar;
   