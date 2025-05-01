import React, { useState, useEffect } from "react";
import axios from "../services/api";
import { useNavigate } from "react-router-dom";

const Auth = ({ setUser }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({ name: "", email: "", password: "" });
    const [message, setMessage] = useState("");
    const navigate = useNavigate();

    useEffect(() => {
        const checkSession = async () => {
            try {
                const response = await axios.get("/session");
                if (response.data.user) {
                    const user = response.data.user;
                    setUser({
                        id: user.id,
                        name: user.name,
                        is_admin: user.is_admin
                    });
                    navigate(user.is_admin ? "/admin" : "/client");
                }
            } catch (error) {
                console.error("No active session found.");
            }
        };
        checkSession();
    }, [navigate, setUser]);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const endpoint = isLogin ? "/login" : "/register";
            const response = await axios.post(endpoint, formData);
            setMessage(response.data.message);
            setUser({
                id: response.data.user_id,
                name: response.data.name,
                is_admin: response.data.is_admin
            });

            // Redirect based on user type
            navigate(response.data.is_admin ? "/admin" : "/client");
        } catch (error) {
            setMessage(error.response?.data?.error || "An error occurred");
        }
    };

    return (
        <div className="h-screen flex flex-col items-center justify-center">
            <h1 className="text-2xl font-bold">{isLogin ? "Login" : "Sign Up"}</h1>
            <form className="flex flex-col gap-2 p-5" onSubmit={handleSubmit}>
                {!isLogin && (
                    <input type="text" name="name" placeholder="Name" value={formData.name} onChange={handleChange} required />
                )}
                <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} required />
                <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} required />
                <button type="submit" className="bg-blue-500 text-white p-2 rounded">
                    {isLogin ? "Login" : "Sign Up"}
                </button>
                <p className="text-sm cursor-pointer text-blue-600" onClick={() => setIsLogin(!isLogin)}>
                    {isLogin ? "Don't have an account? Sign up" : "Already have an account? Login"}
                </p>
            </form>
            {message && <p className="text-red-500">{message}</p>}
        </div>
    );
};

export default Auth;
