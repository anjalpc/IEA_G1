import {
  useEffect,
  useState,
} from "react";

import {
  backendWebSocketUrl,
  fetchDashboardState,
} from "../lib/dashboardApi";

import type {
  DashboardState,
  StateMessage,
} from "../lib/dashboardApi";


type ConnectionState =
  | "connecting"
  | "connected"
  | "disconnected";


export function useDashboardState() {
  const [state, setState] =
    useState<DashboardState | null>(
      null,
    );

  const [connection, setConnection] =
    useState<ConnectionState>(
      "connecting",
    );


  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | null = null;
    let reconnectTimer:
      number | null = null;


    async function loadInitialState() {
      try {
        const initial =
          await fetchDashboardState();

        if (!disposed) {
          setState(initial);
        }
      } catch (error) {
        console.warn(
          "Dashboard initial state unavailable:",
          error,
        );
      }
    }


    function connect() {
      if (disposed) {
        return;
      }

      setConnection(
        "connecting",
      );

      socket =
        new WebSocket(
          backendWebSocketUrl(),
        );


      socket.onopen = () => {
        if (disposed) {
          return;
        }

        setConnection(
          "connected",
        );
      };


      socket.onmessage = (
        event,
      ) => {
        try {
          const message =
            JSON.parse(
              event.data,
            ) as StateMessage;

          if (
            message.type === "state"
          ) {
            setState(
              message.data,
            );
          }
        } catch (error) {
          console.warn(
            "Invalid dashboard WebSocket message:",
            error,
          );
        }
      };


      socket.onclose = () => {
        if (disposed) {
          return;
        }

        setConnection(
          "disconnected",
        );

        reconnectTimer =
          window.setTimeout(
            connect,
            1500,
          );
      };


      socket.onerror = () => {
        socket?.close();
      };
    }


    void loadInitialState();

    connect();


    return () => {
      disposed = true;

      if (
        reconnectTimer !== null
      ) {
        window.clearTimeout(
          reconnectTimer,
        );
      }

      socket?.close();
    };
  }, []);


  return {
    state,
    connection,
  };
}
